from ultralytics import YOLO
import torch, time

def bench(name, ckpt):
    model = YOLO(ckpt).model.eval().half().cuda()
    x = torch.randn(1, 3, 640, 640).half().cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(50):                      # warmup,关键!
            model(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(500):
            model(x)
        torch.cuda.synchronize()
    ms = (time.time() - t) / 500 * 1000
    mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{name:14s} latency={ms:.2f}ms  FPS={1000/ms:.1f}  mem={mem:.0f}MB")

# 跑两遍,顺序对调,看结果是否稳定
bench('YOLOv11n',   '/root/yolo11/utils/yolo11n.pt')
bench('DAF-YOLO-n', '/root/yolo11/utils/DAF-YOLO.pt')
bench('DAF-YOLO-n', '/root/yolo11/utils/DAF-YOLO.pt')
bench('YOLOv11n',   '/root/yolo11/utils/yolo11n.pt')
 # 再跑一次确认稳定
