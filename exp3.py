import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import yaml
import csv

# ========================== 配置 ==========================
WEIGHTS = '/root/yolo11/utils/DAF-YOLO.pt'
YAML_PATH = '/root/yolo11/ultralytics/cfg/datasets/VisDrone.yaml'

IMG_SIZE = 640
IOU_THRES = 0.5
ROI_SCALE = 2.0          # 用 GT 周围 2 倍区域统计 density，更能反映 crowded region
MAX_IMGS = None          # None 表示全量；调试时可设 10
SAVE_CSV = True
OUT_CSV = 'table6_density_recall_results.csv'
# =========================================================


def get_val_image_dir(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    val_path = data.get('val')
    if val_path is None:
        raise ValueError(f"YAML 中未找到 'val' 字段: {yaml_path}")

    val_path = Path(val_path)

    if val_path.is_absolute():
        val_abs = val_path.resolve()
    else:
        dataset_root = data.get('path')
        if dataset_root:
            dataset_root = Path(dataset_root)
            if not dataset_root.is_absolute():
                dataset_root = (Path(yaml_path).parent / dataset_root).resolve()
            val_abs = (dataset_root / val_path).resolve()
        else:
            yaml_dir = Path(yaml_path).parent
            val_abs = (yaml_dir / val_path).resolve()

    if val_abs.exists() and (val_abs / 'images').exists():
        val_abs = val_abs / 'images'

    if not val_abs.exists():
        raise FileNotFoundError(f"验证集图片目录不存在: {val_abs}")

    return val_abs


def image_to_label_path(img_path):
    """
    将 images/val/xxx.jpg 映射到 labels/val/xxx.txt
    适用于 YOLO 格式标签。
    """
    img_path = Path(img_path)
    parts = list(img_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    else:
        label_path = img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt"

    return label_path


def read_yolo_labels(label_path, img_w, img_h):
    """
    读取 YOLO 格式标签:
    cls x_center y_center w h
    返回:
    boxes: [N, 4], xyxy, 原图坐标
    classes: [N]
    """
    boxes, classes = [], []

    if not Path(label_path).exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    with open(label_path, "r") as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 5:
                continue

            cls = int(float(items[0]))
            xc, yc, bw, bh = map(float, items[1:5])

            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h

            x1 = np.clip(x1, 0, img_w - 1)
            y1 = np.clip(y1, 0, img_h - 1)
            x2 = np.clip(x2, 0, img_w - 1)
            y2 = np.clip(y2, 0, img_h - 1)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                classes.append(cls)

    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)


def original_box_to_letterbox_box(box, img_w, img_h, imgsz=640):
    """
    原图 GT 坐标 -> YOLO letterbox 后的 640x640 坐标。
    DASSAFM 的 D/W map 对应的是模型输入，不是原图直接坐标。
    """
    x1, y1, x2, y2 = box

    r = min(imgsz / img_h, imgsz / img_w)
    new_w, new_h = img_w * r, img_h * r
    pad_w = (imgsz - new_w) / 2
    pad_h = (imgsz - new_h) / 2

    lx1 = x1 * r + pad_w
    ly1 = y1 * r + pad_h
    lx2 = x2 * r + pad_w
    ly2 = y2 * r + pad_h

    return np.array([lx1, ly1, lx2, ly2], dtype=np.float32)


def enlarge_box(box, scale, max_w, max_h):
    """
    以 box 中心扩展 scale 倍，用于统计目标周围密度区域。
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = np.clip(cx - bw / 2, 0, max_w - 1)
    ny1 = np.clip(cy - bh / 2, 0, max_h - 1)
    nx2 = np.clip(cx + bw / 2, 0, max_w - 1)
    ny2 = np.clip(cy + bh / 2, 0, max_h - 1)

    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def letterbox_box_to_map_roi(box, map_w, map_h, imgsz=640):
    """
    letterbox 坐标 -> D/W map 坐标。
    """
    x1, y1, x2, y2 = box

    mx1 = int(np.floor(x1 / imgsz * map_w))
    my1 = int(np.floor(y1 / imgsz * map_h))
    mx2 = int(np.ceil(x2 / imgsz * map_w))
    my2 = int(np.ceil(y2 / imgsz * map_h))

    mx1 = int(np.clip(mx1, 0, map_w - 1))
    my1 = int(np.clip(my1, 0, map_h - 1))
    mx2 = int(np.clip(mx2, mx1 + 1, map_w))
    my2 = int(np.clip(my2, my1 + 1, map_h))

    return mx1, my1, mx2, my2


def box_iou_matrix(boxes1, boxes2):
    """
    boxes1: [N,4], boxes2: [M,4]
    返回 IoU: [N,M]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0, boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def match_gt_recall(gt_boxes, gt_cls, pred_boxes, pred_cls, iou_thres=0.5):
    """
    每个 GT 是否被检测到。
    同类别且 IoU >= iou_thres 即认为召回。
    """
    matched = np.zeros(len(gt_boxes), dtype=bool)

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return matched

    ious = box_iou_matrix(gt_boxes, pred_boxes)

    for i in range(len(gt_boxes)):
        same_cls = pred_cls == gt_cls[i]
        if not np.any(same_cls):
            continue

        valid_ious = ious[i, same_cls]
        if len(valid_ious) > 0 and valid_ious.max() >= iou_thres:
            matched[i] = True

    return matched
def compute_gt_quality(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf):
    """
    对每个 GT 计算同类别最佳预测框的定位质量。
    返回 list，每个元素包含:
    best_iou, best_conf, center_error, has_pred
    """
    quality = []

    if len(gt_boxes) == 0:
        return quality

    if len(pred_boxes) == 0:
        for _ in range(len(gt_boxes)):
            quality.append({
                "best_iou": 0.0,
                "best_conf": 0.0,
                "center_error": np.nan,
                "has_pred": False
            })
        return quality

    ious = box_iou_matrix(gt_boxes, pred_boxes)

    for i in range(len(gt_boxes)):
        same_cls_idx = np.where(pred_cls == gt_cls[i])[0]

        if len(same_cls_idx) == 0:
            quality.append({
                "best_iou": 0.0,
                "best_conf": 0.0,
                "center_error": np.nan,
                "has_pred": False
            })
            continue

        same_ious = ious[i, same_cls_idx]
        best_local_idx = int(np.argmax(same_ious))
        best_pred_idx = same_cls_idx[best_local_idx]

        best_iou = float(same_ious[best_local_idx])
        best_conf = float(pred_conf[best_pred_idx])

        gx1, gy1, gx2, gy2 = gt_boxes[i]
        px1, py1, px2, py2 = pred_boxes[best_pred_idx]

        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2

        gt_w = max(gx2 - gx1, 1.0)
        gt_h = max(gy2 - gy1, 1.0)

        # 用 GT 对角线归一化中心误差
        norm = np.sqrt(gt_w ** 2 + gt_h ** 2)
        center_error = float(np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2) / norm)

        quality.append({
            "best_iou": best_iou,
            "best_conf": best_conf,
            "center_error": center_error,
            "has_pred": True
        })

    return quality

def density_bin(d):
    if d < 0.4:
        return "Low-density"
    elif d <= 0.6:
        return "Medium-density"
    else:
        return "High-density"


def patch_dassafm(module):
    """
    替换 DASSAFM forward:
    1. 保存 last_density 和 last_w
    2. 支持 disable_density_bias=True，用于 w/o density bias 对比
    """
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

        if getattr(module, "disable_density_bias", False):
            bias = torch.zeros_like(D)
        else:
            bias = module.k * (D - 0.5)

        logits = torch.cat([l3 + bias, l4, l5 - bias], dim=1)
        w = torch.softmax(logits, dim=1)

        w3 = w[:, 0:1]
        w4 = w[:, 1:2]
        w5 = w[:, 2:3]

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


# ========================== 加载模型 ==========================
model = YOLO(WEIGHTS)
model.model.eval()

dassafm = None
for m in model.model.modules():
    if m.__class__.__name__ == 'DASSAFM':
        dassafm = m
        break

assert dassafm is not None, "未找到 DASSAFM 模块，请检查模型结构或模块名"

patch_dassafm(dassafm)
dassafm.disable_density_bias = False

# ========================== 获取验证集图片 ==========================
img_dir = get_val_image_dir(YAML_PATH)
print(f"验证集图片目录: {img_dir}")

img_paths = []
for suffix in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    img_paths.extend(list(img_dir.glob(suffix)))

img_paths = sorted(img_paths)

if MAX_IMGS is not None:
    img_paths = img_paths[:MAX_IMGS]

print(f"找到 {len(img_paths)} 张图片")

# ========================== 统计容器 ==========================
gt_level_data = []

quality_stats = {
    "Full": {
        "Low-density": {"gt": 0, "best_iou": [], "conf": [], "center_error": [], "r50": 0, "r75": 0},
        "Medium-density": {"gt": 0, "best_iou": [], "conf": [], "center_error": [], "r50": 0, "r75": 0},
        "High-density": {"gt": 0, "best_iou": [], "conf": [], "center_error": [], "r50": 0, "r75": 0},
    },
    "w/o density bias": {
        "Low-density": {"gt": 0, "best_iou": [], "conf": [], "center_error": [], "r50": 0, "r75": 0},
        "Medium-density": {"gt": 0, "best_iou": [], "conf": [], "center_error": [], "r50": 0, "r75": 0},
        "High-density": {"gt": 0, "best_iou": [], "conf": [], "center_error": [], "r50": 0, "r75": 0},
    }
}

skipped_no_label = 0
skipped_no_gt = 0
skipped_roi_error = 0

# ========================== 主循环 ==========================
for img_path in tqdm(img_paths, desc="处理图片"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img_h, img_w = img.shape[:2]

    label_path = image_to_label_path(img_path)
    if not label_path.exists():
        skipped_no_label += 1
        continue

    gt_boxes, gt_cls = read_yolo_labels(label_path, img_w, img_h)

    if len(gt_boxes) == 0:
        skipped_no_gt += 1
        continue

    # ---------- 1) Full DASSAFM: 提取 D/W，并统计 GT-level D/W ----------
    dassafm.disable_density_bias = False

    results = model(
        str(img_path),
        imgsz=IMG_SIZE,
        conf=0.001,
        iou=0.7,
        verbose=False
    )

    if not hasattr(dassafm, "last_density") or not hasattr(dassafm, "last_w"):
        raise RuntimeError("未捕获到 DASSAFM 的 last_density / last_w，请检查 forward patch 是否生效")

    D = dassafm.last_density[0, 0].cpu().numpy()
    w3, w4, w5 = [ww[0, 0].cpu().numpy() for ww in dassafm.last_w]

    map_h, map_w = D.shape

    gt_density_bins = []

    valid_this_image = True

    for box in gt_boxes:
        lb_box = original_box_to_letterbox_box(box, img_w, img_h, IMG_SIZE)
        lb_box = enlarge_box(lb_box, ROI_SCALE, IMG_SIZE, IMG_SIZE)

        x1, y1, x2, y2 = letterbox_box_to_map_roi(lb_box, map_w, map_h, IMG_SIZE)

        roi_D = D[y1:y2, x1:x2]
        roi_w3 = w3[y1:y2, x1:x2]
        roi_w4 = w4[y1:y2, x1:x2]
        roi_w5 = w5[y1:y2, x1:x2]

        if roi_D.size == 0:
            valid_this_image = False
            break

        d_mean = float(np.mean(roi_D))
        w3_mean = float(np.mean(roi_w3))
        w4_mean = float(np.mean(roi_w4))
        w5_mean = float(np.mean(roi_w5))

        b = density_bin(d_mean)
        gt_density_bins.append(b)

        gt_level_data.append({
            "image": str(img_path),
            "bin": b,
            "D": d_mean,
            "W3": w3_mean,
            "W4": w4_mean,
            "W5": w5_mean
        })

    if not valid_this_image or len(gt_density_bins) != len(gt_boxes):
        skipped_roi_error += 1
        continue

    # ---------- 2) Full DASSAFM 的分密度 Recall@0.5 ----------
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        pred_cls = results[0].boxes.cls.cpu().numpy().astype(np.int64)
    else:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_cls = np.zeros((0,), dtype=np.int64)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        pred_cls = results[0].boxes.cls.cpu().numpy().astype(np.int64)
        pred_conf = results[0].boxes.conf.cpu().numpy()
    else:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_cls = np.zeros((0,), dtype=np.int64)
        pred_conf = np.zeros((0,), dtype=np.float32)
    
    quality_full = compute_gt_quality(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
    
    for b, q in zip(gt_density_bins, quality_full):
        quality_stats["Full"][b]["gt"] += 1
        quality_stats["Full"][b]["best_iou"].append(q["best_iou"])
        quality_stats["Full"][b]["conf"].append(q["best_conf"])
    
        if not np.isnan(q["center_error"]):
            quality_stats["Full"][b]["center_error"].append(q["center_error"])
    
        if q["best_iou"] >= 0.5:
            quality_stats["Full"][b]["r50"] += 1
        if q["best_iou"] >= 0.75:
            quality_stats["Full"][b]["r75"] += 1

    # ---------- 3) 关闭 density bias，再跑一次 ----------
    # 注意：这是推理阶段关闭 density bias 的快速验证。
    # 论文最严格的 ablation 最好使用重新训练的 w/o density-bias 权重。
    dassafm.disable_density_bias = True

    results_nobias = model(
        str(img_path),
        imgsz=IMG_SIZE,
        conf=0.001,
        iou=0.7,
        verbose=False
    )

    if results_nobias[0].boxes is not None and len(results_nobias[0].boxes) > 0:
        pred_boxes_nb = results_nobias[0].boxes.xyxy.cpu().numpy()
        pred_cls_nb = results_nobias[0].boxes.cls.cpu().numpy().astype(np.int64)
    else:
        pred_boxes_nb = np.zeros((0, 4), dtype=np.float32)
        pred_cls_nb = np.zeros((0,), dtype=np.int64)

    if results_nobias[0].boxes is not None and len(results_nobias[0].boxes) > 0:
        pred_boxes_nb = results_nobias[0].boxes.xyxy.cpu().numpy()
        pred_cls_nb = results_nobias[0].boxes.cls.cpu().numpy().astype(np.int64)
        pred_conf_nb = results_nobias[0].boxes.conf.cpu().numpy()
    else:
        pred_boxes_nb = np.zeros((0, 4), dtype=np.float32)
        pred_cls_nb = np.zeros((0,), dtype=np.int64)
        pred_conf_nb = np.zeros((0,), dtype=np.float32)
    
    quality_nb = compute_gt_quality(gt_boxes, gt_cls, pred_boxes_nb, pred_cls_nb, pred_conf_nb)
    
    for b, q in zip(gt_density_bins, quality_nb):
        quality_stats["w/o density bias"][b]["gt"] += 1
        quality_stats["w/o density bias"][b]["best_iou"].append(q["best_iou"])
        quality_stats["w/o density bias"][b]["conf"].append(q["best_conf"])
    
        if not np.isnan(q["center_error"]):
            quality_stats["w/o density bias"][b]["center_error"].append(q["center_error"])
    
        if q["best_iou"] >= 0.5:
            quality_stats["w/o density bias"][b]["r50"] += 1
        if q["best_iou"] >= 0.75:
            quality_stats["w/o density bias"][b]["r75"] += 1

# 恢复默认状态
dassafm.disable_density_bias = False

# ========================== 输出 Table 6(a) ==========================
bins_order = ["Low-density", "Medium-density", "High-density"]

print("\n" + "=" * 110)
print("Table 6(a) GT-level statistics of the DASSAFM density-response mechanism")
print("=" * 110)
print(f"{'Density range':<18} {'#GT':<8} {'Mean D':<16} {'Mean W3':<16} {'Mean W4':<16} {'Mean W5':<16}")
print("-" * 110)

table6a_rows = []

for b in bins_order:
    items = [x for x in gt_level_data if x["bin"] == b]
    if len(items) == 0:
        continue

    d_vals = np.array([x["D"] for x in items])
    w3_vals = np.array([x["W3"] for x in items])
    w4_vals = np.array([x["W4"] for x in items])
    w5_vals = np.array([x["W5"] for x in items])

    row = {
        "Density range": b,
        "#GT": len(items),
        "Mean D": d_vals.mean(),
        "Std D": d_vals.std(),
        "Mean W3": w3_vals.mean(),
        "Std W3": w3_vals.std(),
        "Mean W4": w4_vals.mean(),
        "Std W4": w4_vals.std(),
        "Mean W5": w5_vals.mean(),
        "Std W5": w5_vals.std(),
    }
    table6a_rows.append(row)

    print(
        f"{b:<18} "
        f"{len(items):<8d} "
        f"{d_vals.mean():.3f}±{d_vals.std():.3f}     "
        f"{w3_vals.mean():.3f}±{w3_vals.std():.3f}     "
        f"{w4_vals.mean():.3f}±{w4_vals.std():.3f}     "
        f"{w5_vals.mean():.3f}±{w5_vals.std():.3f}"
    )

print("=" * 110)

# ========================== 输出 Table 6(b) ==========================
print("\n" + "=" * 130)
# ========================== 输出 Table 6(b) ==========================
print("\n" + "=" * 130)
print("Table 6(b) Density-stratified localization quality analysis")
print("=" * 130)
print(
    f"{'Density range':<18} "
    f"{'Model':<20} "
    f"{'#GT':<8} "
    f"{'Best IoU':<12} "
    f"{'R@0.5':<10} "
    f"{'R@0.75':<10} "
    f"{'Center Error':<15} "
    f"{'Confidence':<12}"
)
print("-" * 130)

table6b_rows = []

for b in bins_order:
    for model_name in ["w/o density bias", "Full"]:
        s = quality_stats[model_name][b]
        gt_num = s["gt"]

        best_iou = float(np.mean(s["best_iou"])) if len(s["best_iou"]) > 0 else 0.0
        conf = float(np.mean(s["conf"])) if len(s["conf"]) > 0 else 0.0
        center_error = float(np.mean(s["center_error"])) if len(s["center_error"]) > 0 else np.nan

        r50 = s["r50"] / gt_num if gt_num > 0 else 0.0
        r75 = s["r75"] / gt_num if gt_num > 0 else 0.0

        table6b_rows.append({
            "Density range": b,
            "Model": model_name,
            "#GT": gt_num,
            "Best IoU": best_iou,
            "R@0.5": r50,
            "R@0.75": r75,
            "Center Error": center_error,
            "Confidence": conf
        })

        ce_text = f"{center_error:.3f}" if not np.isnan(center_error) else "-"

        print(
            f"{b:<18} "
            f"{model_name:<20} "
            f"{gt_num:<8d} "
            f"{best_iou:<12.3f} "
            f"{r50:<10.3f} "
            f"{r75:<10.3f} "
            f"{ce_text:<15} "
            f"{conf:<12.3f}"
        )

    # 输出 Full - w/o 的差值
    nb = quality_stats["w/o density bias"][b]
    fu = quality_stats["Full"][b]

    nb_gt = nb["gt"]
    fu_gt = fu["gt"]

    nb_iou = float(np.mean(nb["best_iou"])) if len(nb["best_iou"]) > 0 else 0.0
    fu_iou = float(np.mean(fu["best_iou"])) if len(fu["best_iou"]) > 0 else 0.0

    nb_r50 = nb["r50"] / nb_gt if nb_gt > 0 else 0.0
    fu_r50 = fu["r50"] / fu_gt if fu_gt > 0 else 0.0

    nb_r75 = nb["r75"] / nb_gt if nb_gt > 0 else 0.0
    fu_r75 = fu["r75"] / fu_gt if fu_gt > 0 else 0.0

    nb_ce = float(np.mean(nb["center_error"])) if len(nb["center_error"]) > 0 else np.nan
    fu_ce = float(np.mean(fu["center_error"])) if len(fu["center_error"]) > 0 else np.nan

    nb_conf = float(np.mean(nb["conf"])) if len(nb["conf"]) > 0 else 0.0
    fu_conf = float(np.mean(fu["conf"])) if len(fu["conf"]) > 0 else 0.0

    ce_gain = fu_ce - nb_ce if not np.isnan(fu_ce) and not np.isnan(nb_ce) else np.nan
    ce_gain_text = f"{ce_gain:+.3f}" if not np.isnan(ce_gain) else "-"

    table6b_rows.append({
        "Density range": b,
        "Model": "Delta Full - w/o",
        "#GT": "",
        "Best IoU": fu_iou - nb_iou,
        "R@0.5": fu_r50 - nb_r50,
        "R@0.75": fu_r75 - nb_r75,
        "Center Error": ce_gain,
        "Confidence": fu_conf - nb_conf
    })

    print(
        f"{'':<18} "
        f"{'Delta Full - w/o':<20} "
        f"{'':<8} "
        f"{fu_iou - nb_iou:+.3f}       "
        f"{fu_r50 - nb_r50:+.3f}      "
        f"{fu_r75 - nb_r75:+.3f}      "
        f"{ce_gain_text:<15} "
        f"{fu_conf - nb_conf:+.3f}"
    )
    print("-" * 130)

print("=" * 130)