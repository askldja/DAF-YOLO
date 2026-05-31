import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import yaml

# ========== 配置 ==========
WEIGHTS = '/root/yolo11/utils/DAF-YOLO.pt'
YAML_PATH = '/root/yolo11/ultralytics/cfg/datasets/VisDrone.yaml'  # 改为您的绝对路径
STEP = 4
MAX_IMGS = None   # None 表示全量，可先设 10 测试
# ==========================

# 从 YAML 读取 val 图片目录
def get_val_image_dir(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    val_path = data.get('val')
    if val_path is None:
        raise ValueError(f"YAML 中未找到 'val' 字段: {yaml_path}")

    val_path = Path(val_path)

    # 1. 如果 val 本身是绝对路径,直接用
    if val_path.is_absolute():
        val_abs = val_path.resolve()
    else:
        # 2. 优先使用 YAML 中的 'path' 字段作为数据集根目录(Ultralytics 标准格式)
        dataset_root = data.get('path')
        if dataset_root:
            dataset_root = Path(dataset_root)
            if not dataset_root.is_absolute():
                # 'path' 是相对路径,则相对于 YAML 所在目录解析
                dataset_root = (Path(yaml_path).parent / dataset_root).resolve()
            val_abs = (dataset_root / val_path).resolve()
        else:
            # 3. 没有 'path' 字段,退回到相对于 YAML 目录解析
            yaml_dir = Path(yaml_path).parent
            val_abs = (yaml_dir / val_path).resolve()

    # 4. 如果目录存在但下面还有 images/ 子目录,自动进入
    if val_abs.exists() and (val_abs / 'images').exists():
        val_abs = val_abs / 'images'

    if not val_abs.exists():
        raise FileNotFoundError(f"验证集图片目录不存在: {val_abs}")

    return val_abs
# 加载模型
model = YOLO(WEIGHTS)
model.model.eval()

# 查找 DASSAFM 模块
dassafm = None
for m in model.model.modules():
    if m.__class__.__name__ == 'DASSAFM':
        dassafm = m
        break
assert dassafm is not None, "未找到 DASSAFM 模块，请检查模型"
# ========== ⭐ patch 放在这里 ⭐ ==========
def patch_dassafm(module):
    def new_forward(x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 3
            p3, p4, p5 = x
        else:
            p3, p4, p5 = torch.split(x, [module.c3, module.c4, module.c5], dim=1)

        p4p = module.proj4(p4)
        p5p = module.proj5(p5)
        H, W = p3.shape[2], p3.shape[3]
        p4p_up = torch.nn.functional.interpolate(p4p, size=(H, W), mode='nearest')
        p5p_up = torch.nn.functional.interpolate(p5p, size=(H, W), mode='nearest')

        dens_feat = module.density_stem(p3 + p4p_up + p5p_up)
        D = torch.sigmoid(module.density_pred(dens_feat))

        l3 = module.logit3(p3)
        l4 = module.logit4(p4p_up)
        l5 = module.logit5(p5p_up)

        bias = module.k * (D - 0.5)
        logits = torch.cat([l3 + bias, l4, l5 - bias], dim=1)
        w = torch.softmax(logits, dim=1)
        w3, w4, w5 = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        # ⭐ 保存中间结果到模块属性
        module.last_density = D.detach()
        module.last_w = (w3.detach(), w4.detach(), w5.detach())

        fused = w3 * p3 + w4 * p4p_up + w5 * p5p_up
        if module.use_refine:
            fused = module.channel_refine(fused)
            fused = fused * module.spatial_attn(fused)
            enh_p3 = p3 + module.out_conv(fused)
        else:
            enh_p3 = p3 + fused
        return enh_p3

    module.forward = new_forward

patch_dassafm(dassafm)
# ==========================================
# 获取验证集图片目录
img_dir = get_val_image_dir(YAML_PATH)
print(f"验证集图片目录: {img_dir}")

# 获取所有图片
img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
if MAX_IMGS:
    img_paths = img_paths[:MAX_IMGS]
print(f"找到 {len(img_paths)} 张图片")

all_data = []  # (density, (w3,w4,w5))

for img_path in tqdm(img_paths, desc="处理图片"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    # 推理
    results = model(str(img_path), verbose=False)
    # 获取密度图和权重（根据您的实际属性名，这里使用 last_density 和 last_w）
    D = dassafm.last_density[0, 0].cpu().numpy()
    w3, w4, w5 = [w[0, 0].cpu().numpy() for w in dassafm.last_w]
    h, w = D.shape
    for y in range(0, h, STEP):
        for x in range(0, w, STEP):
            d_val = D[y, x]
            if np.isnan(d_val):
                continue
            all_data.append((d_val, (w3[y, x], w4[y, x], w5[y, x])))

print(f"共采集 {len(all_data)} 个像素点")

# 分箱统计
bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
labels = ['低密度 (D<0.4)', '中密度 (0.4≤D≤0.6)', '高密度 (D>0.6)']
results = []
for (low, high), label in zip(bins, labels):
    d_vals, w3_vals, w4_vals, w5_vals = [], [], [], []
    for d, (w3, w4, w5) in all_data:
        if low <= d < high:
            d_vals.append(d)
            w3_vals.append(w3)
            w4_vals.append(w4)
            w5_vals.append(w5)
    if len(d_vals) == 0:
        continue
    results.append({
        'region': label,
        'D_mean': np.mean(d_vals),
        'D_std': np.std(d_vals),
        'w3_mean': np.mean(w3_vals),
        'w3_std': np.std(w3_vals),
        'w4_mean': np.mean(w4_vals),
        'w4_std': np.std(w4_vals),
        'w5_mean': np.mean(w5_vals),
        'w5_std': np.std(w5_vals),
        'count': len(d_vals)
    })

# 打印表格
print("\n" + "="*90)
print("表9  DASSAFM密度感知机制区域级统计（按密度分箱）")
print("="*90)
print(f"{'区域类型':<20} {'D 均值/标准差':<20} {'w₃ 均值/标准差':<20} {'w₄ 均值/标准差':<20} {'w₅ 均值/标准差':<20}")
print("-"*90)
for r in results:
    print(f"{r['region']:<20} {r['D_mean']:.3f}/{r['D_std']:.3f}   "
          f"{r['w3_mean']:.3f}/{r['w3_std']:.3f}   "
          f"{r['w4_mean']:.3f}/{r['w4_std']:.3f}   "
          f"{r['w5_mean']:.3f}/{r['w5_std']:.3f}")
print("="*90)
print(f"总采样点数: {sum(r['count'] for r in results)}")