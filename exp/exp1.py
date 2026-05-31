#!/usr/bin/env python3
"""
validate_density.py  —  Section 5 (density-mechanism validation) for DAF-YOLO.

Produces three sets of results requested by the editor:
  5.1  Spearman correlation between the learned density response D and the
       ground-truth object count, plus a density-stratified D / W3 / W4 / W5
       table (the upgraded Table 6).
  5.2  Density-stratified mAP@0.5 / mAP@0.5:0.95 / recall, DAF-YOLO vs YOLOv11n.
  5.3  Size-stratified COCO AP_small / AP_medium / AP_large + AR_small.

NOTE: You do NOT need to edit your DASSAFM module. D and the fusion weights
(w3,w4,w5) are reconstructed on the fly via forward hooks on density_pred /
logit3 / logit4 / logit5, exactly reproducing the module's forward logic:
    D      = sigmoid(density_pred(...))
    bias   = k * (D - 0.5)
    logits = [l3 + bias, l4, l5 - bias]
    w      = softmax(logits, dim=1)

Install once:
    pip install ultralytics scipy pycocotools pyyaml pillow

Typical run:
    python validate_density.py \
        --weights /path/to/daf-yolo.pt \
        --baseline /path/to/yolov11n_visdrone.pt
"""

import os, glob, json, csv, argparse
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Dataset config (from the user's data yaml)
# ----------------------------------------------------------------------------
DEFAULT_ROOT = "/root/autodl-fs/visdrone_yolo"
NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
         5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"}
NC = len(NAMES)
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")


def list_images(img_dir):
    files = []
    for e in IMG_EXTS:
        files += glob.glob(os.path.join(img_dir, e))
    return sorted(set(files))


def gt_count(stem, lbl_dir):
    """Number of valid GT objects. YOLO-format labels are already filtered
    (ignored/others removed during conversion), so we just count non-empty lines."""
    p = os.path.join(lbl_dir, stem + ".txt")
    if not os.path.exists(p):
        return 0
    return sum(1 for line in open(p) if line.strip())


def find_dassafm(model):
    for m in model.modules():
        if m.__class__.__name__ == "DASSAFM":
            return m
    raise RuntimeError("DASSAFM module not found inside the model.")


# ----------------------------------------------------------------------------
# 5.1  collect per-image mean(D) and mean(w3,w4,w5) via hooks
# ----------------------------------------------------------------------------
def collect_per_image(yolo, img_dir, lbl_dir, imgsz, device):
    dassafm = find_dassafm(yolo.model)
    store, hooks = {}, []
    for nm, sub in [("dp", dassafm.density_pred), ("l3", dassafm.logit3),
                    ("l4", dassafm.logit4), ("l5", dassafm.logit5)]:
        hooks.append(sub.register_forward_hook(
            lambda m, i, o, nm=nm: store.__setitem__(nm, o.detach())))
    k = float(dassafm.k.detach().cpu())

    rows = []
    imgs = list_images(img_dir)
    print(f"[5.1] scanning {len(imgs)} val images ...")
    for j, ip in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        yolo.predict(ip, imgsz=imgsz, device=device, verbose=False)
        D = torch.sigmoid(store["dp"])
        bias = k * (D - 0.5)
        logits = torch.cat([store["l3"] + bias, store["l4"], store["l5"] - bias], dim=1)
        w = torch.softmax(logits, dim=1)
        rows.append(dict(stem=stem, path=ip, gt=gt_count(stem, lbl_dir),
                         meanD=float(D.mean()),
                         w3=float(w[:, 0].mean()),
                         w4=float(w[:, 1].mean()),
                         w5=float(w[:, 2].mean())))
        if (j + 1) % 100 == 0:
            print(f"   {j + 1}/{len(imgs)}")
    for h in hooks:
        h.remove()
    return rows


def report_51(rows, out_dir):
    from scipy.stats import spearmanr
    gt = np.array([r["gt"] for r in rows], dtype=float)
    md = np.array([r["meanD"] for r in rows], dtype=float)
    r, p = spearmanr(md, gt)
    print("\n========== 5.1  D vs ground-truth density ==========")
    print(f"Spearman r(meanD, GT count) = {r:.3f}   (p = {p:.1e}, n = {len(rows)})")

    q1, q2 = np.quantile(gt, [1 / 3, 2 / 3])
    binof = lambda c: 0 if c <= q1 else (1 if c <= q2 else 2)
    names = ["Sparse", "Medium", "Dense"]
    print(f"(tertile thresholds on GT count: q1={q1:.0f}, q2={q2:.0f})")
    header = ["Bin", "n", "MeanGT", "MeanD", "W3", "W4", "W5"]
    print("{:8s}{:>5}{:>9}{:>9}{:>9}{:>9}{:>9}".format(*header))
    table = []
    for b in range(3):
        s = [x for x in rows if binof(x["gt"]) == b]
        f = lambda key: float(np.mean([x[key] for x in s])) if s else 0.0
        rec = [names[b], len(s), round(f("gt"), 1), round(f("meanD"), 3),
               round(f("w3"), 3), round(f("w4"), 3), round(f("w5"), 3)]
        table.append(rec)
        print("{:8s}{:>5}{:>9}{:>9}{:>9}{:>9}{:>9}".format(*[str(x) for x in rec]))

    with open(os.path.join(out_dir, "table6_density_vs_weight.csv"), "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow([f"Spearman_r={r:.3f}", f"p={p:.1e}"])
        wcsv.writerow(header)
        wcsv.writerows(table)
    return rows, binof, names


# ----------------------------------------------------------------------------
# 5.2  density-stratified val (mAP / recall) for both models
# ----------------------------------------------------------------------------
def report_52(rows, binof, names, root, weights, baseline, imgsz, out_dir):
    bin_dir = os.path.join(out_dir, "bins")
    os.makedirs(bin_dir, exist_ok=True)
    yamls = {}
    for b, name in enumerate(["sparse", "medium", "dense"]):
        paths = [x["path"] for x in rows if binof(x["gt"]) == b]
        listf = os.path.abspath(os.path.join(bin_dir, f"val_{name}.txt"))
        with open(listf, "w") as f:
            f.write("\n".join(paths))
        cfg = dict(path=root, train=listf, val=listf, names=NAMES)
        yp = os.path.join(bin_dir, f"visdrone_{name}.yaml")
        yaml.safe_dump(cfg, open(yp, "w"))
        yamls[name] = yp

    print("\n========== 5.2  density-stratified accuracy ==========")
    print("{:8s}{:12s}{:>9}{:>9}{:>9}".format("Bin", "Model", "mAP50", "mAP", "Recall"))
    out_rows = []
    for name in ["sparse", "medium", "dense"]:
        for tag, ck in [("YOLOv11n", baseline), ("DAF-YOLO", weights)]:
            m = YOLO(ck).val(data=yamls[name], imgsz=imgsz, conf=0.001, iou=0.7,
                             max_det=300, verbose=False)
            rec = [name, tag, round(m.box.map50, 4), round(m.box.map, 4), round(m.box.mr, 4)]
            out_rows.append(rec)
            print("{:8s}{:12s}{:>9}{:>9}{:>9}".format(*[str(x) for x in rec]))

    with open(os.path.join(out_dir, "table_density_stratified.csv"), "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["Bin", "Model", "mAP50", "mAP", "Recall"])
        wcsv.writerows(out_rows)


# ----------------------------------------------------------------------------
# 5.3  size-stratified COCO AP (build GT once, evaluate each model)
# ----------------------------------------------------------------------------
def build_coco_gt(img_dir, lbl_dir, gt_path):
    imgs = list_images(img_dir)
    id_of = {os.path.splitext(os.path.basename(p))[0]: i + 1 for i, p in enumerate(imgs)}
    images, anns, aid = [], [], 1
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        w, h = Image.open(p).size
        images.append(dict(id=id_of[stem], file_name=os.path.basename(p), width=w, height=h))
        lp = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lp):
            continue
        for line in open(lp):
            t = line.split()
            if len(t) < 5:
                continue
            c, cx, cy, bw, bh = map(float, t[:5])
            x, y, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            anns.append(dict(id=aid, image_id=id_of[stem], category_id=int(c) + 1,
                             bbox=[x, y, ww, hh], area=ww * hh, iscrowd=0))
            aid += 1
    json.dump(dict(images=images, annotations=anns,
                   categories=[dict(id=i + 1, name=NAMES[i]) for i in range(NC)]),
              open(gt_path, "w"))
    return id_of


def run_predictions(ck, img_dir, id_of, imgsz, device, dt_path):
    yolo = YOLO(ck)
    dets = []
    for p in list_images(img_dir):
        stem = os.path.splitext(os.path.basename(p))[0]
        r = yolo.predict(p, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                         device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, cf, cl):
            dets.append(dict(image_id=id_of[stem], category_id=int(c) + 1,
                             bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             score=float(s)))
    json.dump(dets, open(dt_path, "w"))


def report_53(img_dir, lbl_dir, weights, baseline, imgsz, device, out_dir):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt_path = os.path.join(out_dir, "gt_val.json")
    id_of = build_coco_gt(img_dir, lbl_dir, gt_path)

    print("\n========== 5.3  size-stratified COCO AP ==========")
    print("{:12s}{:>10}{:>10}{:>10}{:>10}".format("Model", "AP_small", "AP_med", "AP_large", "AR_small"))
    out_rows = []
    for tag, ck in [("YOLOv11n", baseline), ("DAF-YOLO", weights)]:
        dt_path = os.path.join(out_dir, f"dt_{tag}.json")
        run_predictions(ck, img_dir, id_of, imgsz, device, dt_path)
        cg = COCO(gt_path)
        cd = cg.loadRes(dt_path)
        E = COCOeval(cg, cd, "bbox")
        # For very crowded VisDrone scenes you may uncomment the next line to raise
        # the detection cap from the COCO default of 100 to 500:
        # E.params.maxDets = [1, 10, 500]
        E.evaluate(); E.accumulate(); E.summarize()
        rec = [tag, round(E.stats[3], 4), round(E.stats[4], 4),
               round(E.stats[5], 4), round(E.stats[9], 4)]
        out_rows.append(rec)
        print("{:12s}{:>10}{:>10}{:>10}{:>10}".format(*[str(x) for x in rec]))

    with open(os.path.join(out_dir, "table_size_stratified.csv"), "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["Model", "AP_small", "AP_medium", "AP_large", "AR_small"])
        wcsv.writerows(out_rows)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="VisDrone YOLO dataset root")
    ap.add_argument("--weights", required=True, help="trained DAF-YOLO checkpoint (.pt)")
    ap.add_argument("--baseline", required=True, help="trained YOLOv11n checkpoint (.pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="density_eval_out")
    ap.add_argument("--parts", default="all",
                    help="comma list among {51,52,53} or 'all'")
    args = ap.parse_args()

    img_dir = os.path.join(args.root, "images", "val")
    lbl_dir = os.path.join(args.root, "labels", "val")
    os.makedirs(args.out, exist_ok=True)
    parts = ["51", "52", "53"] if args.parts == "all" else args.parts.split(",")

    rows = binof = names = None
    if "51" in parts or "52" in parts:
        yolo = YOLO(args.weights)
        rows = collect_per_image(yolo, img_dir, lbl_dir, args.imgsz, args.device)
        json.dump(rows, open(os.path.join(args.out, "per_image.json"), "w"))
        rows, binof, names = report_51(rows, args.out)

    if "52" in parts:
        report_52(rows, binof, names, args.root, args.weights, args.baseline,
                  args.imgsz, args.out)

    if "53" in parts:
        report_53(img_dir, lbl_dir, args.weights, args.baseline,
                  args.imgsz, args.device, args.out)

    print(f"\nAll CSV tables written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()