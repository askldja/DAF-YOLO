import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

def collect_region_stats(model, img_dir, device='cuda', img_size=640, step=4, max_imgs=None):
    """遍历图片文件夹，收集密度图和权重"""
    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png')) + list(Path(img_dir).glob('*.JPG'))
    if max_imgs is not None:
        img_paths = img_paths[:max_imgs]
    print(f"找到 {len(img_paths)} 张图片")

    model.to(device)
    model.eval()

    # 自动查找 DASSAFM 模块
    dassafm = None
    for module in model.model.model.modules():
        if hasattr(module, 'last_density') and hasattr(module, 'last_weights'):
            dassafm = module
            break
    if dassafm is None:
        raise RuntimeError("未找到 DASSAFM 模块！请确认模型中已添加 last_density 和 last_weights 属性。")

    all_data = []  # (density, (w3,w4,w5))

    for img_path in tqdm(img_paths, desc="处理图片"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, (img_size, img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().div(255).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model.model(img_tensor)
            D = dassafm.last_density[0, 0].cpu().numpy()       # (H,W)
            weights = dassafm.last_weights[0].cpu().numpy()    # (3,H,W) -> w3,w4,w5

        h, w = D.shape
        for y in range(0, h, step):
            for x in range(0, w, step):
                d_val = D[y, x]
                if np.isnan(d_val):
                    continue
                w3_val = weights[0, y, x]
                w4_val = weights[1, y, x]
                w5_val = weights[2, y, x]
                all_data.append((d_val, (w3_val, w4_val, w5_val)))

    return all_data

def compute_region_stats(data):
    bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
    labels = ['低密度 (D<0.4)', '中密度 (0.4≤D≤0.6)', '高密度 (D>0.6)']
    results = []
    for (low, high), label in zip(bins, labels):
        d_vals, w3_vals, w4_vals, w5_vals = [], [], [], []
        for d, (w3, w4, w5) in data:
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
    return results

def print_table(results):
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

def plot_weights_vs_density(results):
    categories = [r['region'] for r in results]
    w3 = [r['w3_mean'] for r in results]
    w4 = [r['w4_mean'] for r in results]
    w5 = [r['w5_mean'] for r in results]
    x = np.arange(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - width, w3, width, label='w₃ (浅层)', color='#1f77b4')
    ax.bar(x, w4, width, label='w₄ (中层)', color='#ff7f0e')
    ax.bar(x + width, w5, width, label='w₅ (深层)', color='#2ca02c')
    ax.set_ylabel('平均融合权重', fontsize=12)
    ax.set_xlabel('密度区域', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('density_weight_trend.png', dpi=300)
    plt.show()
    print("趋势图已保存为 density_weight_trend.png")

if __name__ == '__main__':
    # ========== 请修改为您的实际路径 ==========
    model_path = '/root/yolo11/runs/detect/11n gai/weights/best.pt'
    # 请将下方路径改为您实际的 VisDrone 验证集图片目录
    img_dir = '/root/autodl-fs/visdrone_yolo/images'
    # ==========================================

    # 加载模型
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"验证集图片目录: {img_dir}")

    # 检查目录是否存在
    if not Path(img_dir).exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")

    # 收集数据（可设置 max_imgs=10 先测试）
    all_data = collect_region_stats(model, img_dir, device, step=4, max_imgs=None)

    # 统计并输出表格
    results = compute_region_stats(all_data)
    print_table(results)
    plot_weights_vs_density(results)