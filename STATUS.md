# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-26 — Two more from-scratch builds: (1) backprop + Adam from scratch (cnn/autograd.py — hand-derived gradients for conv/pool/relu/linear/softmax-CE, im2col, gradient-checked vs finite diff) → train the CNN with ZERO autograd (train_scratch.py, pure NumPy, 99.7% val in ~5s); (2) Kalman filter from scratch (kalman.py — general KF + CVKalman1D + TargetKalman3D, a drop-in for Estimator3D with covariance/uncertainty; verified the fire-control loop hits using it). Added a "Built from first principles" table to the README. 132 tests green.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
