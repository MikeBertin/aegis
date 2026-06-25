# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-25 — M5: from-scratch CNN target discriminator. Own conv net (src/aegis/cnn/ — our architecture + training loop) trained on synthetic patches to ID the designated red balloon vs distractors (99% val acc, CPU). "From scratch" made literal: conv/pool/linear re-implemented in pure NumPy (cnn/conv.py, hand-computed tests) and proven to match PyTorch's forward to ~1e-7 on trained weights. Learned discriminator gates targeting (confirms designated target before fireable). train_cnn.py CLI + cnn_viz.py README image. 116 tests (torch-gated ones skip without torch).
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
