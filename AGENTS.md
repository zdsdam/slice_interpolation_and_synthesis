# SynthSR Replication MVP

Goal:
HR MRI → sparse-slice simulation → interpolation → 3D residual U-Net
→ evaluation → NIfTI → 3D Slicer.

Implementation order:
1. src/sparse_simulator.py
2. src/interpolation.py
3. src/data_loader.py
4. src/preprocessing.py
5. src/models/blocks.py
6. src/models/unet3d.py
7. src/losses.py
8. src/train.py
9. src/metrics.py
10. src/evaluate.py
11. src/inference.py
12. src/export_nifti.py

Current milestone: modules 1–4 only.

Rules:
- Keep this an MVP.
- Use NumPy for volume operations.
- Use nibabel for NIfTI I/O.
- Use PyTorch for deep learning.
- Prefer small functions and simple APIs.
- Use type hints and concise docstrings.
- Add focused tests for each module.
- Do not implement later modules early.
- Preserve MRI spatial metadata where relevant.