# STATUS — Project AEGIS

State: 🟢 ACTIVE
Phase: 3 — BUILD (M3 software done; hardware-gated)
Last action: 2026-06-23 — M3 actuation layer + safety gate built & mock-tested. SafetyGate (track-all/fire-inanimate-only, hard denylist, lock + person-interlock), hardware driver ABCs + servo mapping + mock drivers + lazy PCA9685/Nerf stubs, Turret integration wired into live pipeline (--turret mock|real). BOM at docs/HARDWARE.md. 42 headless tests green.
Next action: ORDER the M3 kit (~£425, see docs/HARDWARE.md / HARDWARE-BUDGET.md). Meanwhile optionally start M4 dataset work, or eyeball the live mock loop: `python main.py --classes "sports ball" --turret mock`.
Blocked by: Hardware not yet ordered (Jetson + gimbal + servos + PCA9685 + blaster). Everything up to "drive real servos" is built.
Next milestone: M3 physical bring-up — gimbal tracks under real servos (darts out), then gated firing
Outcome: —

> *Update MEMORY.md when significant milestones are reached or the project closes.*
