# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-24 — M4 dataset→train→export pipeline built & smoke-tested end-to-end (synthetic balloons → build → 1-epoch YOLOv11 fine-tune → ONNX export, on CPU). aegis.data (pure label/split/data.yaml, tested) + capture.py/train.py/export.py + synth generator. 'balloon' added to SafetyGate fireable allowlist. 51 headless tests green. Docs: M4-TRAINING.md.
Next action: ORDER the M3 kit (~£425, docs/HARDWARE.md) to unblock physical build. M4 real-data path (capture→label→train) can also start now on the laptop — needs a webcam + a labelling pass.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
