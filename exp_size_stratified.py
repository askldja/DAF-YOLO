import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import yaml


# ========================== 配置 ==========================
YAML_PATH = '/root/yolo11/ultralytics/cfg/datasets/VisDrone.yaml'

MODEL_LIST = {
    'YOLOv11n': '/root/yolo11/utils/yolo11n.pt',
    'YOLOv5s' :'/root/yolo11/runs/detect/V5S/weights/best.pt',
    'YOLOv8n': '/root/yolo11/runs/detect/v8n/weights/best.pt',
    'YOLOv8s': '/root/yolo11/runs/detect/v8s/weights/best.pt',
    'YOLOv9s': '/root/yolo11/runs/detect/v9s/weights/best.pt',
    'YOLOvv10n': '/root/yolo11/runs/detect/v10n/weights/best.pt',
    'DAF-YOLO-n': '/root/yolo11/utils/DAF-YOLO.pt',
    'DAF-YOLO-s': '/root/yolo11/runs/detect/v11s+改进/weights/best.pt',
}

IMG_SIZE = 640
CONF_THRES = 0.001
NMS_IOU = 0.7
USE_HALF = True
DEVICE = 0

MAX_IMGS = None   # None 表示全量；调试可设 20
SAVE_CSV = True
OUT_CSV = 'object_size_stratified_results.csv'

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
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
    """
    读取 YOLO 标签:
    cls xc yc w h
    返回 boxes xyxy 原图坐标、classes、area
    """
    boxes, classes, areas = [], [], []

    if not Path(label_path).exists():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

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

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)

            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
            areas.append(area)

    return (
        np.array(boxes, dtype=np.float32),
        np.array(classes, dtype=np.int64),
        np.array(areas, dtype=np.float32),
    )


def size_bin_from_area(area):
    if area < 32 * 32:
        return 'small'
    elif area < 96 * 96:
        return 'medium'
    else:
        return 'large'


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


def compute_ap(recall, precision):
    """
    COCO 风格 101-point AP.
    """
    if len(recall) == 0:
        return 0.0

    recall_points = np.linspace(0, 1, 101)
    ap = 0.0

    for r in recall_points:
        inds = np.where(recall >= r)[0]
        p = np.max(precision[inds]) if inds.size > 0 else 0.0
        ap += p / 101.0

    return float(ap)


def collect_dataset_gts(img_paths):
    """
    收集整个验证集 GT。
    返回:
    gts[image_id] = {
        'boxes': ...,
        'cls': ...,
        'size': np.array(['small', 'medium', 'large'])
    }
    """
    gts = {}
    class_set = set()

    for img_id, img_path in enumerate(tqdm(img_paths, desc="读取 GT")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        label_path = image_to_label_path(img_path)

        boxes, classes, areas = read_yolo_labels(label_path, img_w, img_h)
        sizes = np.array([size_bin_from_area(a) for a in areas])

        for c in classes:
            class_set.add(int(c))

        gts[img_id] = {
            'path': img_path,
            'boxes': boxes,
            'cls': classes,
            'size': sizes,
        }

    return gts, sorted(list(class_set))


def collect_model_predictions(model_name, weight_path, img_paths):
    """
    收集某个模型在所有图片上的预测结果。
    返回 preds[image_id] = {
        'boxes': ...,
        'cls': ...,
        'conf': ...
    }
    """
    print("\n" + "=" * 100)
    print(f"开始推理: {model_name}")
    print(f"权重路径: {weight_path}")
    print("=" * 100)

    if not Path(weight_path).exists():
        raise FileNotFoundError(f"权重不存在: {weight_path}")

    model = YOLO(weight_path)

    try:
        model.fuse()
    except Exception as e:
        print(f"model.fuse() 跳过: {e}")

    preds = {}

    for img_id, img_path in enumerate(tqdm(img_paths, desc=f"{model_name} 推理")):
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
            confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
            confs = np.zeros((0,), dtype=np.float32)

        preds[img_id] = {
            'boxes': boxes,
            'cls': classes,
            'conf': confs,
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds


def evaluate_size_bin(gts, preds, class_list, size_name, iou_thresholds):
    """
    计算某个 size bin 的 AP50、AP75、AP50:95、AR50、AR75、AR50:95。
    为避免不同尺寸目标互相干扰：
    - 当前 size 的 GT 作为有效 GT；
    - 如果预测框匹配到同类但非当前 size 的 GT，则忽略该预测，不计 FP。
    """
    ap_by_thr = []
    ar_by_thr = []

    ap50 = None
    ap75 = None
    ar50 = None
    ar75 = None

    for thr in iou_thresholds:
        class_aps = []
        class_ars = []

        for cls_id in class_list:
            npos = 0

            # 每张图当前 class + 当前 size 的 GT
            gt_valid = {}
            gt_ignore = {}

            for img_id, gt in gts.items():
                boxes = gt['boxes']
                classes = gt['cls']
                sizes = gt['size']

                valid_mask = (classes == cls_id) & (sizes == size_name)
                ignore_mask = (classes == cls_id) & (sizes != size_name)

                valid_boxes = boxes[valid_mask]
                ignore_boxes = boxes[ignore_mask]

                gt_valid[img_id] = {
                    'boxes': valid_boxes,
                    'matched': np.zeros(len(valid_boxes), dtype=bool),
                }
                gt_ignore[img_id] = ignore_boxes

                npos += len(valid_boxes)

            if npos == 0:
                continue

            # 收集当前类别所有预测
            pred_records = []
            for img_id, pr in preds.items():
                boxes = pr['boxes']
                classes = pr['cls']
                confs = pr['conf']

                mask = classes == cls_id
                for box, conf in zip(boxes[mask], confs[mask]):
                    pred_records.append({
                        'img_id': img_id,
                        'box': box,
                        'conf': float(conf),
                    })

            if len(pred_records) == 0:
                class_aps.append(0.0)
                class_ars.append(0.0)
                continue

            pred_records = sorted(pred_records, key=lambda x: x['conf'], reverse=True)

            tp = []
            fp = []

            for rec in pred_records:
                img_id = rec['img_id']
                pbox = rec['box'][None, :]

                valid_boxes = gt_valid[img_id]['boxes']
                ignore_boxes = gt_ignore[img_id]

                # 先尝试匹配当前 size 的 GT
                if len(valid_boxes) > 0:
                    ious = box_iou_matrix(pbox, valid_boxes)[0]
                    best_idx = int(np.argmax(ious))
                    best_iou = float(ious[best_idx])
                else:
                    best_idx = -1
                    best_iou = 0.0

                if best_iou >= thr and best_idx >= 0 and not gt_valid[img_id]['matched'][best_idx]:
                    tp.append(1)
                    fp.append(0)
                    gt_valid[img_id]['matched'][best_idx] = True
                    continue

                # 如果该预测明显对应其他 size 的同类 GT，则忽略，不计 FP
                ignore_this_pred = False
                if len(ignore_boxes) > 0:
                    ignore_ious = box_iou_matrix(pbox, ignore_boxes)[0]
                    if float(ignore_ious.max()) >= thr:
                        ignore_this_pred = True

                if ignore_this_pred:
                    continue

                tp.append(0)
                fp.append(1)

            if len(tp) == 0:
                class_aps.append(0.0)
                class_ars.append(0.0)
                continue

            tp = np.array(tp)
            fp = np.array(fp)

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)

            recall = cum_tp / max(npos, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

            ap = compute_ap(recall, precision)
            ar = float(recall[-1]) if len(recall) > 0 else 0.0

            class_aps.append(ap)
            class_ars.append(ar)

        mean_ap = float(np.mean(class_aps)) if len(class_aps) > 0 else 0.0
        mean_ar = float(np.mean(class_ars)) if len(class_ars) > 0 else 0.0

        ap_by_thr.append(mean_ap)
        ar_by_thr.append(mean_ar)

        if abs(thr - 0.50) < 1e-6:
            ap50 = mean_ap
            ar50 = mean_ar
        if abs(thr - 0.75) < 1e-6:
            ap75 = mean_ap
            ar75 = mean_ar

    result = {
        'AP50': ap50 if ap50 is not None else 0.0,
        'AP75': ap75 if ap75 is not None else 0.0,
        'AP50_95': float(np.mean(ap_by_thr)) if len(ap_by_thr) > 0 else 0.0,
        'AR50': ar50 if ar50 is not None else 0.0,
        'AR75': ar75 if ar75 is not None else 0.0,
        'AR50_95': float(np.mean(ar_by_thr)) if len(ar_by_thr) > 0 else 0.0,
    }

    return result


def count_gt_by_size(gts):
    counts = {'small': 0, 'medium': 0, 'large': 0}
    for gt in gts.values():
        for s in gt['size']:
            counts[s] += 1
    return counts


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

    gts, class_list = collect_dataset_gts(img_paths)
    size_counts = count_gt_by_size(gts)

    print("\nGT size distribution:")
    print(f"small : {size_counts['small']}")
    print(f"medium: {size_counts['medium']}")
    print(f"large : {size_counts['large']}")
    print(f"class list: {class_list}")

    all_results = []

    for model_name, weight_path in MODEL_LIST.items():
        preds = collect_model_predictions(model_name, weight_path, img_paths)

        model_result = {'Model': model_name}

        for size_name in ['small', 'medium', 'large']:
            res = evaluate_size_bin(
                gts=gts,
                preds=preds,
                class_list=class_list,
                size_name=size_name,
                iou_thresholds=IOU_THRESHOLDS
            )

            prefix = size_name.capitalize()

            model_result[f'{prefix}_AP50'] = res['AP50'] * 100
            model_result[f'{prefix}_AP75'] = res['AP75'] * 100
            model_result[f'{prefix}_AP50_95'] = res['AP50_95'] * 100
            model_result[f'{prefix}_AR50'] = res['AR50'] * 100
            model_result[f'{prefix}_AR75'] = res['AR75'] * 100
            model_result[f'{prefix}_AR50_95'] = res['AR50_95'] * 100

        all_results.append(model_result)

    # ========================== 输出主表 ==========================
    print("\n" + "=" * 140)
    print("Object-size-stratified performance on VisDrone2019 validation set")
    print("=" * 140)
    print(
        f"{'Model':<14} "
        f"{'APs':<8} {'APm':<8} {'APl':<8} "
        f"{'ARs':<8} {'ARm':<8} {'ARl':<8} "
        f"{'APs@50':<8} {'ARs@50':<8}"
    )
    print("-" * 140)

    for r in all_results:
        print(
            f"{r['Model']:<14} "
            f"{r['Small_AP50_95']:<8.2f} "
            f"{r['Medium_AP50_95']:<8.2f} "
            f"{r['Large_AP50_95']:<8.2f} "
            f"{r['Small_AR50_95']:<8.2f} "
            f"{r['Medium_AR50_95']:<8.2f} "
            f"{r['Large_AR50_95']:<8.2f} "
            f"{r['Small_AP50']:<8.2f} "
            f"{r['Small_AR50']:<8.2f}"
        )

    print("=" * 140)

    # 如果正好有 YOLOv11n 和 DAF-YOLO-n，输出 gain
    names = [r['Model'] for r in all_results]
    if 'YOLOv11n' in names and 'DAF-YOLO-n' in names:
        base = all_results[names.index('YOLOv11n')]
        ours = all_results[names.index('DAF-YOLO-n')]

        print("\nGain: DAF-YOLO-n - YOLOv11n")
        print("-" * 100)
        print(f"AP_small gain   : {ours['Small_AP50_95'] - base['Small_AP50_95']:+.2f}")
        print(f"AP_medium gain  : {ours['Medium_AP50_95'] - base['Medium_AP50_95']:+.2f}")
        print(f"AP_large gain   : {ours['Large_AP50_95'] - base['Large_AP50_95']:+.2f}")
        print(f"AR_small gain   : {ours['Small_AR50_95'] - base['Small_AR50_95']:+.2f}")
        print(f"AR_medium gain  : {ours['Medium_AR50_95'] - base['Medium_AR50_95']:+.2f}")
        print(f"AR_large gain   : {ours['Large_AR50_95'] - base['Large_AR50_95']:+.2f}")
        print(f"AP_small@50 gain: {ours['Small_AP50'] - base['Small_AP50']:+.2f}")
        print(f"AR_small@50 gain: {ours['Small_AR50'] - base['Small_AR50']:+.2f}")

    # ========================== 保存 CSV ==========================
    if SAVE_CSV:
        fieldnames = [
            'Model',
            'Small_AP50', 'Small_AP75', 'Small_AP50_95',
            'Small_AR50', 'Small_AR75', 'Small_AR50_95',
            'Medium_AP50', 'Medium_AP75', 'Medium_AP50_95',
            'Medium_AR50', 'Medium_AR75', 'Medium_AR50_95',
            'Large_AP50', 'Large_AP75', 'Large_AP50_95',
            'Large_AR50', 'Large_AR75', 'Large_AR50_95',
        ]

        with open(OUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        print(f"\n结果已保存到: {OUT_CSV}")


if __name__ == '__main__':
    main()