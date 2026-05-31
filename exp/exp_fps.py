import csv
import torch
from pathlib import Path
from ultralytics import YOLO

# ========================== 配置 ==========================
YAML_PATH = '/root/yolo11/ultralytics/cfg/datasets/VisDrone.yaml'

MODEL_LIST = {
    'YOLOv11n': '/root/yolo11/utils/yolo11n.pt',
    # 'YOLOv5s' :'/root/yolo11/runs/detect/V5S/weights/best.pt',
    # 'YOLOv8n': '/root/yolo11/runs/detect/v8n/weights/best.pt',
    # 'YOLOv8s': '/root/yolo11/runs/detect/v8s/weights/best.pt',
    # 'YOLOv9s': '/root/yolo11/runs/detect/v9s/weights/best.pt',
    # 'YOLOvv10n': '/root/yolo11/runs/detect/v10n/weights/best.pt',
    '+s': '/root/yolo11/runs/detect/11n+eva/weights/best.pt',
    '+d': '/root/yolo11/runs/detect/yolo11E+DASSAFM/weights/best.pt',

    
    'DAF-YOLO-n': '/root/yolo11/utils/DAF-YOLO.pt',
    # 'DAF-YOLO-s': '/root/yolo11/runs/detect/v11s+改进/weights/best.pt',
}

IMG_SIZE = 640
BATCH_SIZE = 1
DEVICE = 0
USE_HALF = True

SAVE_CSV = True
OUT_CSV = 'fps_val_speed_results.csv'
# =========================================================


def safe_get_speed(metrics):
    """
    Ultralytics val 返回的 metrics.speed 通常包含:
    preprocess / inference / loss / postprocess
    单位是 ms/image
    """
    speed = getattr(metrics, 'speed', None)

    if speed is None:
        return {
            'preprocess': 0.0,
            'inference': 0.0,
            'postprocess': 0.0,
            'total': 0.0,
        }

    preprocess = float(speed.get('preprocess', 0.0))
    inference = float(speed.get('inference', 0.0))
    postprocess = float(speed.get('postprocess', 0.0))

    total = preprocess + inference + postprocess

    return {
        'preprocess': preprocess,
        'inference': inference,
        'postprocess': postprocess,
        'total': total,
    }


def safe_get_map(metrics):
    """
    如果需要顺便保存 mAP，也可以从 metrics.box 中取。
    不同 ultralytics 版本字段基本一致。
    """
    box = getattr(metrics, 'box', None)

    if box is None:
        return {
            'P': None,
            'R': None,
            'mAP50': None,
            'mAP50_95': None,
        }

    return {
        'P': float(getattr(box, 'mp', 0.0)) * 100,
        'R': float(getattr(box, 'mr', 0.0)) * 100,
        'mAP50': float(getattr(box, 'map50', 0.0)) * 100,
        'mAP50_95': float(getattr(box, 'map', 0.0)) * 100,
    }


def benchmark_one_model(model_name, weight_path):
    print("\n" + "=" * 100)
    print(f"开始测试: {model_name}")
    print(f"权重路径: {weight_path}")
    print("=" * 100)

    if not Path(weight_path).exists():
        print(f"跳过 {model_name}: 权重不存在 -> {weight_path}")
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(weight_path)

    try:
        model.fuse()
        print("已执行 model.fuse()")
    except Exception as e:
        print(f"model.fuse() 跳过: {e}")

    metrics = model.val(
        data=YAML_PATH,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        half=USE_HALF,
        plots=False,
        save=False,
        save_json=False,
        verbose=False,
    )

    speed = safe_get_speed(metrics)
    maps = safe_get_map(metrics)

    inference_ms = speed['inference']
    total_ms = speed['total']

    inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    total_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        device_name = torch.cuda.get_device_name(0)
    else:
        peak_memory = 0.0
        device_name = 'CPU'

    result = {
        'Model': model_name,
        'Weights': weight_path,
        'Input': IMG_SIZE,
        'Batch': BATCH_SIZE,
        'Precision': 'FP16' if USE_HALF else 'FP32',
        'Preprocess_ms': speed['preprocess'],
        'Inference_ms': speed['inference'],
        'Postprocess_ms': speed['postprocess'],
        'Total_ms': speed['total'],
        'Inference_FPS': inference_fps,
        'Total_FPS': total_fps,
        'Peak_memory_MB': peak_memory,
        'Device': device_name,
        'P': maps['P'],
        'R': maps['R'],
        'mAP50': maps['mAP50'],
        'mAP50_95': maps['mAP50_95'],
    }

    print("\n测试结果:")
    print(f"Model:             {result['Model']}")
    print(f"Device:            {result['Device']}")
    print(f"Precision:         {result['Precision']}")
    print(f"Input size:         {IMG_SIZE}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Preprocess:         {result['Preprocess_ms']:.3f} ms/image")
    print(f"Inference:          {result['Inference_ms']:.3f} ms/image")
    print(f"Postprocess:        {result['Postprocess_ms']:.3f} ms/image")
    print(f"Total:              {result['Total_ms']:.3f} ms/image")
    print(f"Inference FPS:      {result['Inference_FPS']:.2f}")
    print(f"Total FPS:          {result['Total_FPS']:.2f}")
    print(f"Peak GPU memory:    {result['Peak_memory_MB']:.1f} MB")
    print(f"mAP@0.5:            {result['mAP50']:.2f}")
    print(f"mAP@0.5:0.95:       {result['mAP50_95']:.2f}")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    all_results = []

    for model_name, weight_path in MODEL_LIST.items():
        result = benchmark_one_model(model_name, weight_path)
        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 150)
    print("Ultralytics val speed results")
    print("=" * 150)
    print(
        f"{'Model':<16} "
        f"{'Precision':<10} "
        f"{'Pre/ms':<10} "
        f"{'Infer/ms':<10} "
        f"{'Post/ms':<10} "
        f"{'Total/ms':<10} "
        f"{'Infer FPS':<12} "
        f"{'Total FPS':<12} "
        f"{'Mem/MB':<10} "
        f"{'mAP50':<8} "
        f"{'mAP50-95':<10}"
    )
    print("-" * 150)

    for r in all_results:
        print(
            f"{r['Model']:<16} "
            f"{r['Precision']:<10} "
            f"{r['Preprocess_ms']:<10.3f} "
            f"{r['Inference_ms']:<10.3f} "
            f"{r['Postprocess_ms']:<10.3f} "
            f"{r['Total_ms']:<10.3f} "
            f"{r['Inference_FPS']:<12.2f} "
            f"{r['Total_FPS']:<12.2f} "
            f"{r['Peak_memory_MB']:<10.1f} "
            f"{r['mAP50']:<8.2f} "
            f"{r['mAP50_95']:<10.2f}"
        )

    print("=" * 150)

    if SAVE_CSV:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Model',
                'Weights',
                'Input',
                'Batch',
                'Precision',
                'Preprocess_ms',
                'Inference_ms',
                'Postprocess_ms',
                'Total_ms',
                'Inference_FPS',
                'Total_FPS',
                'Peak_memory_MB',
                'Device',
                'P',
                'R',
                'mAP50',
                'mAP50_95',
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        print(f"\n结果已保存到: {OUT_CSV}")


if __name__ == "__main__":
    main()