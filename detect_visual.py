import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


# ========================== 配置 ==========================
WEIGHTS = "/root/yolo11/utils/DAF-YOLO.pt"

# 可以是单张图片，也可以是图片文件夹
SOURCE = "/root/yolo11/utils/test"

# 输出文件夹
SAVE_DIR = "/root/yolo11/utils/failure_visual_results"

IMG_SIZE = 640
CONF_THRES = 0.25
NMS_IOU = 0.7
DEVICE = 0
USE_HALF = True

# TP / FP / FN 判断 IoU 阈值
MATCH_IOU = 0.5

# 是否只保存存在漏检或误检的图片
SAVE_FAILURE_ONLY = False

# 最多处理多少张图片；None 表示全量
MAX_IMGS = None
# =========================================================


def image_to_label_path(img_path):
    """
    将 images/val/xxx.jpg 映射到 labels/val/xxx.txt
    如果没有标签文件，则只画预测框。
    """
    img_path = Path(img_path)
    parts = list(img_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    else:
        label_path = img_path.with_suffix(".txt")

    return label_path


def read_yolo_labels(label_path, img_w, img_h):
    """
    YOLO 格式:
    cls x_center y_center w h
    返回原图坐标 xyxy。
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


def match_predictions(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf, iou_thres=0.5):
    """
    匹配预测框和 GT。
    返回:
    tp_pred_indices: 正确检测预测框 index
    fp_pred_indices: 误检预测框 index
    fn_gt_indices: 漏检 GT index
    """
    tp_pred_indices = []
    fp_pred_indices = []
    matched_gt = set()

    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))

    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    ious = box_iou_matrix(pred_boxes, gt_boxes)

    # 按置信度从高到低匹配
    order = np.argsort(-pred_conf)

    for pi in order:
        same_cls = gt_cls == pred_cls[pi]

        if not np.any(same_cls):
            fp_pred_indices.append(pi)
            continue

        candidate_gt_indices = np.where(same_cls)[0]
        candidate_ious = ious[pi, candidate_gt_indices]

        best_local = int(np.argmax(candidate_ious))
        best_gt = int(candidate_gt_indices[best_local])
        best_iou = float(candidate_ious[best_local])

        if best_iou >= iou_thres and best_gt not in matched_gt:
            tp_pred_indices.append(pi)
            matched_gt.add(best_gt)
        else:
            fp_pred_indices.append(pi)

    fn_gt_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]

    return tp_pred_indices, fp_pred_indices, fn_gt_indices


def draw_box(img, box, color, text=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        t = max(1, thickness)

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, t)
        y_text = max(y1 - 5, th + 5)

        cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), color, -1)
        cv2.putText(img, text, (x1 + 2, y_text - 2), font, font_scale, (255, 255, 255), t, cv2.LINE_AA)


def get_image_paths(source):
    source = Path(source)

    if source.is_file():
        return [source]

    img_paths = []
    for suffix in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(list(source.glob(suffix)))

    return sorted(img_paths)


def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_paths = get_image_paths(SOURCE)

    if MAX_IMGS is not None:
        img_paths = img_paths[:MAX_IMGS]

    print(f"找到 {len(img_paths)} 张图片")
    print(f"保存目录: {save_dir}")

    model = YOLO(WEIGHTS)

    try:
        model.fuse()
    except Exception as e:
        print(f"model.fuse() 跳过: {e}")

    names = model.names

    total_saved = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in tqdm(img_paths, desc="检测图片"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

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
            pred_boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_cls = results[0].boxes.cls.cpu().numpy().astype(np.int64)
            pred_conf = results[0].boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_cls = np.zeros((0,), dtype=np.int64)
            pred_conf = np.zeros((0,), dtype=np.float32)

        label_path = image_to_label_path(img_path)
        gt_boxes, gt_cls = read_yolo_labels(label_path, img_w, img_h)

        vis = img.copy()

        # 有标签：画 TP / FP / FN
        if label_path.exists():
            tp_idx, fp_idx, fn_idx = match_predictions(
                gt_boxes=gt_boxes,
                gt_cls=gt_cls,
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                pred_conf=pred_conf,
                iou_thres=MATCH_IOU
            )

            total_tp += len(tp_idx)
            total_fp += len(fp_idx)
            total_fn += len(fn_idx)

            has_failure = len(fp_idx) > 0 or len(fn_idx) > 0

            if SAVE_FAILURE_ONLY and not has_failure:
                continue

            # TP: 绿色
            for pi in tp_idx:
                cls_id = int(pred_cls[pi])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                text = f"TP {cls_name} {pred_conf[pi]:.2f}"
                draw_box(vis, pred_boxes[pi], color=(0, 180, 0), text=text, thickness=2)

            # FP: 黄色
            for pi in fp_idx:
                cls_id = int(pred_cls[pi])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                text = f"FP {cls_name} {pred_conf[pi]:.2f}"
                draw_box(vis, pred_boxes[pi], color=(0, 220, 255), text=text, thickness=2)

            # FN: 红色
            for gi in fn_idx:
                cls_id = int(gt_cls[gi])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                text = f"FN {cls_name}"
                draw_box(vis, gt_boxes[gi], color=(0, 0, 255), text=text, thickness=2)

        # 没有标签：只画预测框
        else:
            for box, cls_id, conf in zip(pred_boxes, pred_cls, pred_conf):
                cls_id = int(cls_id)
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                text = f"{cls_name} {conf:.2f}"
                draw_box(vis, box, color=(0, 180, 0), text=text, thickness=2)

        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), vis)
        total_saved += 1

    print("\n完成")
    print(f"保存图片数: {total_saved}")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")
    print(f"结果目录: {save_dir}")


if __name__ == "__main__":
    main()