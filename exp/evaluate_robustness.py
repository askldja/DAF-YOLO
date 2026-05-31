"""
DAF-YOLO 鲁棒性实验:批量评测脚本
依次评测 6 种扰动 × 2 个模型(YOLOv11n / DAF-YOLO)

使用方法:
    1. 先用 perturbations.py 生成扰动图像目录
    2. 修改本文件中的路径变量
    3. python evaluate_robustness.py
"""

from ultralytics import YOLO
import os
import json

# ============ 修改这些路径为你本地的实际路径 ============

# 模型权重路径
MODEL_PATHS = {
    'YOLOv11n':  'yolo11n.pt',                       # 基线
    'DAF-YOLO':  'runs/train/daf-yolo-n/best.pt',    # 你训好的 DAF-YOLO
}

# 数据集 YAML 文件根目录 (会自动为每种扰动创建对应 YAML)
PERTURBED_ROOT = '/path/to/VisDrone2019-DET-val-perturbed'

# VisDrone 原始 YAML(参考它的结构创建扰动版的)
BASE_YAML = 'datasets/VisDrone.yaml'

# ============ 6 种扰动配置 ============
PERTURBATIONS = [
    'clean',       # 清晰图基线
    'gauss_005',   # 高斯噪声 σ=0.05
    'gauss_010',   # 高斯噪声 σ=0.10
    'jpeg_q50',    # JPEG 压缩 Q=50
    'jpeg_q30',    # JPEG 压缩 Q=30
    'fog_005',     # 加雾 β=0.05
    'fog_010',     # 加雾 β=0.10
]


def make_perturbed_yaml(perturbation_name):
    """
    为扰动数据集创建临时 YAML 配置(基于 VisDrone.yaml 修改 val 路径)
    """
    import yaml
    with open(BASE_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    if perturbation_name == 'clean':
        return BASE_YAML
    cfg['val'] = os.path.join(PERTURBED_ROOT, perturbation_name, 'images')
    tmp_yaml = f'datasets/VisDrone_{perturbation_name}.yaml'
    with open(tmp_yaml, 'w') as f:
        yaml.dump(cfg, f)
    return tmp_yaml


def main():
    results = {}  # {model_name: {perturbation: {mAP50, mAP50-95, ...}}}

    for model_name, weight_path in MODEL_PATHS.items():
        model = YOLO(weight_path)
        results[model_name] = {}

        for pert in PERTURBATIONS:
            print(f'\n===== {model_name} on {pert} =====')
            yaml_path = make_perturbed_yaml(pert)

            metrics = model.val(
                data=yaml_path,
                imgsz=640,
                batch=16,
                device=0,
                save_json=False,
                verbose=False,
                plots=False,
            )

            results[model_name][pert] = {
                'mAP50':      float(metrics.box.map50),
                'mAP50-95':   float(metrics.box.map),
                'precision':  float(metrics.box.mp),
                'recall':     float(metrics.box.mr),
            }
            print(f'  mAP@0.5    = {metrics.box.map50:.4f}')
            print(f'  mAP@0.5:95 = {metrics.box.map:.4f}')

    # 保存结果
    with open('robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\n所有结果已保存到 robustness_results.json')

    # 生成结果表格
    print_table(results)


def print_table(results):
    """打印格式化的结果表格"""
    print('\n' + '=' * 100)
    print(f"{'扰动类型':<20} {'模型':<12} {'mAP@0.5':<10} {'mAP@0.5:95':<12} {'退化率(%)':<10}")
    print('=' * 100)

    for pert in PERTURBATIONS:
        for model_name in MODEL_PATHS.keys():
            r = results[model_name][pert]
            base = results[model_name]['clean']
            if pert == 'clean':
                deg = '—'
            else:
                deg = f"{(r['mAP50'] - base['mAP50']) / base['mAP50'] * 100:+.1f}%"
            print(f"{pert:<20} {model_name:<12} {r['mAP50']*100:<10.1f} "
                  f"{r['mAP50-95']*100:<12.1f} {deg:<10}")
        print('-' * 100)


if __name__ == '__main__':
    main()
