import csv
import cv2
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


# ========================== 配置 ==========================
YAML_PATH = '/root/yolo11/ultralytics/cfg/datasets/VisDrone.yaml'

MODEL_LIST = {

    'YOLOv8n': '/root/yolo11/runs/detect/v8n/weights/best.pt',
    # 'YOLOv8s': '/root/yolo11/runs/detect/v8s/weights/best.pt',
    # 'YOLOv9s': '/root/yolo11/runs/detect/v9s/weights/best.pt',
    'YOLOvv10n': '/root/yolo11/runs/detect/v10n/weights/best.pt',
    'YOLOv11n': '/root/yolo11/utils/yolo11n.pt',
    'DAF-YOLO-n': '/root/yolo11/utils/DAF-YOLO.pt',
    # 'DAF-YOLO-s': '/root/yolo11/runs/detect/v11s+改进/weights/best.pt',
}

IMG_SIZE = 640
CONF_THRES = 0.001
NMS_IOU = 0.7
DEVICE = 0
USE_HALF = True

# 在 640x640 letterbox 坐标系下统计邻居
# 如果 Crowded 样本太少，可以改成 48 或 64
CROWD_RADIUS = 48

MAX_IMGS = None
SAVE_CSV = True
OUT_CSV = 'crowded_region_recall_results.csv'
# =========================================================


def get_val_image_dir(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    val_path = data.get('val')
    if val_path is None:
        raise ValueError(f"YAML 中未找到 val 字段: {yaml_path}")

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
            val_abs = (Path(yaml_path).parent / val_path).resolve()

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

    with open(label_path, 'r') as f:
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
    原图坐标 -> 640x640 letterbox 坐标。
    用于在统一尺度下计算目标邻居数量。
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


def compute_crowd_bins(gt_boxes, img_w, img_h, imgsz=640, radius=48):
    """
    对每个 GT 计算其中心点 radius 范围内的其他 GT 数量。
    在 letterbox 640x640 坐标系中计算，避免原图尺寸差异影响。
    """
    n = len(gt_boxes)

    if n == 0:
        return [], np.zeros((0,), dtype=np.int64)

    lb_boxes = np.array([
        original_box_to_letterbox_box(box, img_w, img_h, imgsz)
        for box in gt_boxes
    ], dtype=np.float32)

    centers = np.zeros((n, 2), dtype=np.float32)
    centers[:, 0] = (lb_boxes[:, 0] + lb_boxes[:, 2]) / 2
    centers[:, 1] = (lb_boxes[:, 1] + lb_boxes[:, 3]) / 2

    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # 排除自己
    neighbor_count = ((dist <= radius) & (dist > 0)).sum(axis=1).astype(np.int64)

    bins = []
    for c in neighbor_count:
        if c == 0:
            bins.append("Sparse")
        elif c <= 3:
            bins.append("Moderate")
        else:
            bins.append("Crowded")

    return bins, neighbor_count


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


def get_predictions(model, img_path):
    results = model.predict(
        source=str(img_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=NMS_IOU,
        device=DEVICE,
        half=USE_HALF,
        verbose=False
    )

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        classes = results[0].boxes.cls.cpu().numpy().astype(np.int64)
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int64)

    return boxes, classes


def init_stats():
    return {
        "Sparse": {"gt": 0, "best_iou": [], "r50": 0, "r75": 0, "miss50": 0},
        "Moderate": {"gt": 0, "best_iou": [], "r50": 0, "r75": 0, "miss50": 0},
        "Crowded": {"gt": 0, "best_iou": [], "r50": 0, "r75": 0, "miss50": 0},
    }


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    img_dir = get_val_image_dir(YAML_PATH)
    print(f"验证集图片目录: {img_dir}")

    img_paths = []
    for suffix in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_paths.extend(list(Path(img_dir).glob(suffix)))

    img_paths = sorted(img_paths)

    if MAX_IMGS is not None:
        img_paths = img_paths[:MAX_IMGS]

    print(f"找到 {len(img_paths)} 张图片")
    print(f"Crowd radius in 640x640 letterbox space: {CROWD_RADIUS}")

    # 加载模型
    models = {}
    for name, weight_path in MODEL_LIST.items():
        if not Path(weight_path).exists():
            raise FileNotFoundError(f"{name} 权重不存在: {weight_path}")

        print(f"加载模型: {name} -> {weight_path}")
        model = YOLO(weight_path)

        try:
            model.fuse()
        except Exception as e:
            print(f"{name} fuse 跳过: {e}")

        models[name] = model

    crowd_order = ["Sparse", "Moderate", "Crowded"]

    metric_stats = {
        name: init_stats()
        for name in MODEL_LIST.keys()
    }

    neighbor_all = []
    skipped_no_label = 0
    skipped_no_gt = 0

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

        crowd_bins, neighbor_count = compute_crowd_bins(
            gt_boxes=gt_boxes,
            img_w=img_w,
            img_h=img_h,
            imgsz=IMG_SIZE,
            radius=CROWD_RADIUS
        )

        neighbor_all.extend(neighbor_count.tolist())

        for model_name, model in models.items():
            pred_boxes, pred_cls = get_predictions(model, img_path)

            best_ious = compute_gt_best_iou(
                gt_boxes=gt_boxes,
                gt_cls=gt_cls,
                pred_boxes=pred_boxes,
                pred_cls=pred_cls
            )

            for cbin, iou_val in zip(crowd_bins, best_ious):
                metric_stats[model_name][cbin]["gt"] += 1
                metric_stats[model_name][cbin]["best_iou"].append(float(iou_val))

                if iou_val >= 0.5:
                    metric_stats[model_name][cbin]["r50"] += 1
                else:
                    metric_stats[model_name][cbin]["miss50"] += 1

                if iou_val >= 0.75:
                    metric_stats[model_name][cbin]["r75"] += 1

    # ========================== 输出 crowd 分布 ==========================
    print("\n" + "=" * 100)
    print("GT neighbor-count distribution")
    print("=" * 100)

    neighbor_all = np.array(neighbor_all, dtype=np.int64)

    if len(neighbor_all) > 0:
        print(f"Total GT: {len(neighbor_all)}")
        print(f"Mean neighbors: {neighbor_all.mean():.2f}")
        print(f"Median neighbors: {np.median(neighbor_all):.2f}")
        print(f"Max neighbors: {neighbor_all.max()}")
        print(f"Sparse   count: {(neighbor_all == 0).sum()}")
        print(f"Moderate count: {((neighbor_all >= 1) & (neighbor_all <= 3)).sum()}")
        print(f"Crowded  count: {(neighbor_all >= 4).sum()}")

    # ========================== 输出主表 ==========================
    print("\n" + "=" * 125)
    print("Crowded-region recall analysis on VisDrone2019 validation set")
    print("=" * 125)
    print(
        f"{'Crowd level':<14} "
        f"{'Model':<14} "
        f"{'#GT':<8} "
        f"{'Best IoU':<12} "
        f"{'R@0.5':<10} "
        f"{'R@0.75':<10} "
        f"{'Miss@0.5':<10} "
        f"{'Gain R@0.5':<12}"
    )
    print("-" * 125)

    rows = []

    model_names = list(MODEL_LIST.keys())
    base_name = model_names[0]
    ours_name = model_names[1] if len(model_names) > 1 else None

    for cbin in crowd_order:
        base_r50 = None

        for model_name in model_names:
            s = metric_stats[model_name][cbin]
            gt_num = s["gt"]

            best_iou = float(np.mean(s["best_iou"])) if len(s["best_iou"]) > 0 else 0.0
            r50 = s["r50"] / gt_num if gt_num > 0 else 0.0
            r75 = s["r75"] / gt_num if gt_num > 0 else 0.0
            miss50 = s["miss50"]

            if model_name == base_name:
                base_r50 = r50
                gain_text = "-"
            else:
                gain_text = f"{r50 - base_r50:+.3f}" if base_r50 is not None else "-"

            rows.append({
                "Crowd level": cbin,
                "Model": model_name,
                "#GT": gt_num,
                "Best IoU": best_iou,
                "R@0.5": r50,
                "R@0.75": r75,
                "Miss@0.5": miss50,
                "Gain R@0.5": gain_text,
            })

            print(
                f"{cbin:<14} "
                f"{model_name:<14} "
                f"{gt_num:<8d} "
                f"{best_iou:<12.3f} "
                f"{r50:<10.3f} "
                f"{r75:<10.3f} "
                f"{miss50:<10d} "
                f"{gain_text:<12}"
            )

        if ours_name is not None:
            base = metric_stats[base_name][cbin]
            ours = metric_stats[ours_name][cbin]

            base_gt = base["gt"]
            ours_gt = ours["gt"]

            base_iou = float(np.mean(base["best_iou"])) if len(base["best_iou"]) > 0 else 0.0
            ours_iou = float(np.mean(ours["best_iou"])) if len(ours["best_iou"]) > 0 else 0.0

            base_r50 = base["r50"] / base_gt if base_gt > 0 else 0.0
            ours_r50 = ours["r50"] / ours_gt if ours_gt > 0 else 0.0

            base_r75 = base["r75"] / base_gt if base_gt > 0 else 0.0
            ours_r75 = ours["r75"] / ours_gt if ours_gt > 0 else 0.0

            base_miss = base["miss50"]
            ours_miss = ours["miss50"]

            print(
                f"{'':<14} "
                f"{'Delta':<14} "
                f"{'':<8} "
                f"{ours_iou - base_iou:+.3f}       "
                f"{ours_r50 - base_r50:+.3f}      "
                f"{ours_r75 - base_r75:+.3f}      "
                f"{ours_miss - base_miss:+d}        "
                f"{ours_r50 - base_r50:+.3f}"
            )

        print("-" * 125)

    print("=" * 125)

    print("\n运行信息：")
    print(f"总图片数: {len(img_paths)}")
    print(f"无标签文件跳过: {skipped_no_label}")
    print(f"无 GT 跳过: {skipped_no_gt}")

    # ========================== 保存 CSV ==========================
    if SAVE_CSV:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Crowd level",
                "Model",
                "#GT",
                "Best IoU",
                "R@0.5",
                "R@0.75",
                "Miss@0.5",
                "Gain R@0.5",
            ])
            writer.writeheader()

            for r in rows:
                writer.writerow({
                    "Crowd level": r["Crowd level"],
                    "Model": r["Model"],
                    "#GT": r["#GT"],
                    "Best IoU": f"{r['Best IoU']:.6f}",
                    "R@0.5": f"{r['R@0.5']:.6f}",
                    "R@0.75": f"{r['R@0.75']:.6f}",
                    "Miss@0.5": r["Miss@0.5"],
                    "Gain R@0.5": r["Gain R@0.5"],
                })

        print(f"\n结果已保存到: {OUT_CSV}")


if __name__ == "__main__":
    main()