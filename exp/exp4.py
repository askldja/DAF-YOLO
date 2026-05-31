import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import yaml
import csv

# ========================== 配置 ==========================
WEIGHTS_DAF = '/root/yolo11/utils/DAF-YOLO.pt'
WEIGHTS_BASE = '/root/yolo11/utils/yolo11n.pt'   # 改成你的 YOLOv11n 权重路径

YAML_PATH = '/root/yolo11/ultralytics/cfg/datasets/VisDrone.yaml'

IMG_SIZE = 640
ROI_SCALE = 2.0
MAX_IMGS = None

CONF_THRES = 0.001
NMS_IOU = 0.7

SAVE_CSV = True
OUT_CSV = 'table6_daf_vs_yolov11n_density_recall.csv'
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


def compute_gt_best_iou(gt_boxes, gt_cls, pred_boxes, pred_cls):
    """
    对每个 GT 找同类别预测框的最大 IoU。
    返回 best_ious: [N]
    """
    best_ious = np.zeros(len(gt_boxes), dtype=np.float32)

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return best_ious

    ious = box_iou_matrix(gt_boxes, pred_boxes)

    for i in range(len(gt_boxes)):
        same_cls = pred_cls == gt_cls[i]
        if not np.any(same_cls):
            best_ious[i] = 0.0
        else:
            best_ious[i] = float(ious[i, same_cls].max())

    return best_ious


def density_bin(d):
    if d < 0.4:
        return "Low-density"
    elif d <= 0.6:
        return "Medium-density"
    else:
        return "High-density"


def patch_dassafm(module):
    """
    只用于提取 DAF-YOLO 中 DASSAFM 的 density response 和 scale weights。
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


def get_predictions(model, img_path):
    results = model(
        str(img_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=NMS_IOU,
        verbose=False
    )

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        pred_cls = results[0].boxes.cls.cpu().numpy().astype(np.int64)
    else:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_cls = np.zeros((0,), dtype=np.int64)

    return pred_boxes, pred_cls


def init_metric_dict():
    return {
        "Low-density": {"gt": 0, "best_iou": [], "r50": 0, "r75": 0, "miss50": 0},
        "Medium-density": {"gt": 0, "best_iou": [], "r50": 0, "r75": 0, "miss50": 0},
        "High-density": {"gt": 0, "best_iou": [], "r50": 0, "r75": 0, "miss50": 0},
    }


# ========================== 加载模型 ==========================
print("加载 DAF-YOLO...")
daf_model = YOLO(WEIGHTS_DAF)
daf_model.model.eval()

dassafm = None
for m in daf_model.model.modules():
    if m.__class__.__name__ == 'DASSAFM':
        dassafm = m
        break

assert dassafm is not None, "未找到 DAF-YOLO 中的 DASSAFM 模块，请检查模型结构或模块名"
patch_dassafm(dassafm)

print("加载 YOLOv11n baseline...")
base_model = YOLO(WEIGHTS_BASE)
base_model.model.eval()

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
bins_order = ["Low-density", "Medium-density", "High-density"]

gt_level_data = []

metric_stats = {
    "YOLOv11n": init_metric_dict(),
    "DAF-YOLO": init_metric_dict(),
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

    # ---------- 1) 用 DAF-YOLO 提取 D/W，并确定 GT density bin ----------
    pred_boxes_daf, pred_cls_daf = get_predictions(daf_model, img_path)

    if not hasattr(dassafm, "last_density") or not hasattr(dassafm, "last_w"):
        raise RuntimeError("未捕获到 DASSAFM 的 last_density / last_w，请检查 patch 是否生效")

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

    # ---------- 2) 获取 YOLOv11n baseline 预测 ----------
    pred_boxes_base, pred_cls_base = get_predictions(base_model, img_path)

    # ---------- 3) 计算两个模型对同一批 GT 的 best IoU ----------
    best_iou_daf = compute_gt_best_iou(gt_boxes, gt_cls, pred_boxes_daf, pred_cls_daf)
    best_iou_base = compute_gt_best_iou(gt_boxes, gt_cls, pred_boxes_base, pred_cls_base)

    for model_name, best_ious in [
        ("YOLOv11n", best_iou_base),
        ("DAF-YOLO", best_iou_daf),
    ]:
        for b, iou_val in zip(gt_density_bins, best_ious):
            metric_stats[model_name][b]["gt"] += 1
            metric_stats[model_name][b]["best_iou"].append(float(iou_val))

            if iou_val >= 0.5:
                metric_stats[model_name][b]["r50"] += 1
            else:
                metric_stats[model_name][b]["miss50"] += 1

            if iou_val >= 0.75:
                metric_stats[model_name][b]["r75"] += 1


# ========================== 输出 Table 6(a) ==========================
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
print("\n" + "=" * 125)
print("Table 6(b) Density-stratified detection gains over YOLOv11n")
print("=" * 125)
print(
    f"{'Density range':<18} "
    f"{'Model':<12} "
    f"{'#GT':<8} "
    f"{'Best IoU':<12} "
    f"{'R@0.5':<10} "
    f"{'R@0.75':<10} "
    f"{'Miss@0.5':<10} "
    f"{'Gain R@0.5':<12}"
)
print("-" * 125)

table6b_rows = []

for b in bins_order:
    base = metric_stats["YOLOv11n"][b]
    daf = metric_stats["DAF-YOLO"][b]

    base_gt = base["gt"]
    daf_gt = daf["gt"]

    base_iou = float(np.mean(base["best_iou"])) if len(base["best_iou"]) > 0 else 0.0
    daf_iou = float(np.mean(daf["best_iou"])) if len(daf["best_iou"]) > 0 else 0.0

    base_r50 = base["r50"] / base_gt if base_gt > 0 else 0.0
    daf_r50 = daf["r50"] / daf_gt if daf_gt > 0 else 0.0

    base_r75 = base["r75"] / base_gt if base_gt > 0 else 0.0
    daf_r75 = daf["r75"] / daf_gt if daf_gt > 0 else 0.0

    base_miss = base["miss50"]
    daf_miss = daf["miss50"]

    gain_r50 = daf_r50 - base_r50

    rows = [
        ("YOLOv11n", base_gt, base_iou, base_r50, base_r75, base_miss, "-"),
        ("DAF-YOLO", daf_gt, daf_iou, daf_r50, daf_r75, daf_miss, f"{gain_r50:+.3f}"),
    ]

    for model_name, gt_num, best_iou, r50, r75, miss50, gain_text in rows:
        table6b_rows.append({
            "Density range": b,
            "Model": model_name,
            "#GT": gt_num,
            "Best IoU": best_iou,
            "R@0.5": r50,
            "R@0.75": r75,
            "Miss@0.5": miss50,
            "Gain R@0.5": gain_text,
        })

        print(
            f"{b:<18} "
            f"{model_name:<12} "
            f"{gt_num:<8d} "
            f"{best_iou:<12.3f} "
            f"{r50:<10.3f} "
            f"{r75:<10.3f} "
            f"{miss50:<10d} "
            f"{gain_text:<12}"
        )

    # 额外输出 DAF - YOLOv11n 的差值
    print(
        f"{'':<18} "
        f"{'Delta':<12} "
        f"{'':<8} "
        f"{daf_iou - base_iou:+.3f}       "
        f"{daf_r50 - base_r50:+.3f}      "
        f"{daf_r75 - base_r75:+.3f}      "
        f"{daf_miss - base_miss:+d}        "
        f"{gain_r50:+.3f}"
    )
    print("-" * 125)

print("=" * 125)


# ========================== 运行信息 ==========================
print("\n运行信息：")
print(f"总图片数: {len(img_paths)}")
print(f"无标签文件跳过: {skipped_no_label}")
print(f"无 GT 跳过: {skipped_no_gt}")
print(f"ROI 异常跳过: {skipped_roi_error}")
print(f"有效 GT 数: {len(gt_level_data)}")


# ========================== 保存 CSV ==========================
if SAVE_CSV:
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Table 6(a) GT-level statistics of DASSAFM density-response mechanism"])
        writer.writerow([
            "Density range", "#GT",
            "Mean D", "Std D",
            "Mean W3", "Std W3",
            "Mean W4", "Std W4",
            "Mean W5", "Std W5"
        ])

        for r in table6a_rows:
            writer.writerow([
                r["Density range"],
                r["#GT"],
                f"{r['Mean D']:.6f}",
                f"{r['Std D']:.6f}",
                f"{r['Mean W3']:.6f}",
                f"{r['Std W3']:.6f}",
                f"{r['Mean W4']:.6f}",
                f"{r['Std W4']:.6f}",
                f"{r['Mean W5']:.6f}",
                f"{r['Std W5']:.6f}",
            ])

        writer.writerow([])
        writer.writerow(["Table 6(b) Density-stratified detection gains over YOLOv11n"])
        writer.writerow([
            "Density range",
            "Model",
            "#GT",
            "Best IoU",
            "R@0.5",
            "R@0.75",
            "Miss@0.5",
            "Gain R@0.5"
        ])

        for r in table6b_rows:
            writer.writerow([
                r["Density range"],
                r["Model"],
                r["#GT"],
                f"{r['Best IoU']:.6f}",
                f"{r['R@0.5']:.6f}",
                f"{r['R@0.75']:.6f}",
                r["Miss@0.5"],
                r["Gain R@0.5"],
            ])

    print(f"\n结果已保存到: {OUT_CSV}")