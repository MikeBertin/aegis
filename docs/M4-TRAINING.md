# AEGIS — M4: Custom Detector (dataset → train → edge-deploy)

The portfolio-gold milestone: collect and label a dataset, fine-tune YOLOv11 on
a bespoke target ("balloon" — the fireable showcase target), and deploy it
optimised on the Jetson. All of this runs on a laptop except the final
TensorRT engine build, which must happen on the Jetson.

## The pipeline
```
capture.py ──> label (Roboflow/CVAT/labelImg) ──> data.yaml
                                                      │
synth.py ──> build_dataset() ─────────────────────────┤
                                                      ▼
                                              train.py (YOLOv11 fine-tune)
                                                      │  runs/<name>/weights/best.pt
                                                      ▼
                                              export.py ──> ONNX (laptop)
                                                          └─> TensorRT .engine (Jetson)
                                                      │
                                                      ▼
                              main.py --model best.pt --classes balloon --turret mock
```

## 0. Smoke-test the whole pipeline (no real data, no hardware)
Proves build → train → export end-to-end in seconds on CPU:
```bash
python train.py --synthetic 16 --epochs 1 --imgsz 160 --batch 8 --device cpu --name smoke
python export.py runs/smoke/weights/best.pt --format onnx --imgsz 160
```
`synth.py` generates coloured-ellipse "balloons" on noisy backgrounds — enough
to exercise every stage. (Won't detect real balloons; it's plumbing validation,
and a seed for transfer learning.)

## 1. Collect real data
```bash
python capture.py --name balloons --interval 1   # auto-grab every 1s
```
Aim for **150–300 images**: vary distance, angle, lighting, background, and
include frames with people/pets present (so the model learns to *not* fire near
them — the negatives matter). Frames land in `datasets/balloons/raw/`.

## 2. Label
Import `datasets/balloons/raw/` into Roboflow / CVAT / labelImg, draw boxes for
the `balloon` class, export in **YOLOv11** format. You'll get
`images/{train,val}` + `labels/{train,val}` + a `data.yaml`. (The label format
and split logic are exactly what `aegis.data.dataset` implements and tests.)

## 3. Train
```bash
python train.py --data datasets/balloons/data.yaml --epochs 100 --imgsz 640
```
On the laptop CPU this is slow but fine for a first model; on the Jetson (or any
CUDA box) it's much faster. Fine-tunes from `yolo11n.pt`. Watch `mAP50` climb in
the run summary; weights land in `runs/<name>/weights/best.pt` (gitignored).

## 4. Deploy
- **Laptop test:** `python main.py --model runs/<name>/weights/best.pt --classes balloon --turret mock`
- **On the Jetson:** copy `best.pt`, then build the optimised engine *on-device*:
  ```bash
  python export.py best.pt --format engine --half        # FP16, fast
  python export.py best.pt --format engine --int8        # INT8, needs calibration imgs
  ```
  TensorRT `.engine` files are GPU/JetPack-specific — never copy one between
  machines. INT8 wants a few hundred representative frames for calibration.

## Why this is the interesting milestone
- **Edge optimisation is real work, not a library call:** PyTorch → ONNX →
  TensorRT, with FP16/INT8 quantisation and a calibration set — the gap between
  "ran a model" and "deployed a model".
- **The detector closes the safety loop:** `balloon` is on the `SafetyGate`
  fireable allowlist, so a custom-trained class becomes a fireable target while
  people/animals stay on the hard denylist.
