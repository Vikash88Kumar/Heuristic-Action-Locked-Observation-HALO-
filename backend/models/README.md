# Model weights

Place your three trained checkpoints here (not included in this zip --
they're large binary files you already have locally):

- `detection_best.pt`  — YOLO detector (player/ball)
- `jersey_ocr_best.pt` — jersey number CNN
- `ccnn_best.pt`       — possession/touch temporal filter

On Render/Railway, upload these to a persistent disk (see `render.yaml`)
rather than committing them to git -- they're too large for a normal
git repo and `.gitignore` already excludes `models/*.pt`.
