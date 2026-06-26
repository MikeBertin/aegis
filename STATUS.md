# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M1–M4 software all done; hardware-gated)
Last action: 2026-06-26 — Built a THIRD interactive demo (docs/site/cnn.html — ③ Vision/CNN) that runs the real from-scratch NumPy CNN in-browser (Pyodide+NumPy, our cnn/conv.py + exported trained weights): a convolution playground (pick kernel → feature map) + live classifier (draw a balloon → conv-1 feature maps + TARGET verdict; red square scores 0 → learned colour AND shape). Linked all 3 demos via the evolution nav. Verified in browser (red balloon 0.996, others ~0, no console errors). tools/export_cnn.py generates cnn_weights.js + cnn_conv.js.
Next action: Decide when to make the repo public + enable GitHub Pages (unlocks the live demo URL). To unblock the physical build, ORDER the M3 kit (~£425, docs/HARDWARE.md). M4 real-data path (capture→label→train) can start now on the laptop.
Blocked by: Hardware not ordered (M3 physical). All M1–M4 software is built & tested; remaining work is real-world: servos, real dataset, on-Jetson TensorRT.
Next milestone: M3 physical bring-up (gimbal tracks under real servos, darts out) — needs hardware
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
