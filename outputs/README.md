# Hand surface video experiments

- `tracking-surface-demo.mp4`: initial surface approximation.
- `tracking-surface-refined.mp4`: slimmer tapered digits, smoother silhouettes, supersampled rendering, softer palm shading, and subtle illustrative distal nail/joint accents.

Both use the original `../demo.mov`, preserve the 1678 × 788 stereo canvas, and export at 30 fps. The first five seconds retain the recording; a 1.2-second fade introduces the surface.

This is an image-space visualization derived from the colored skeleton pixels. It does not recover a calibrated 3D hand, mesh topology, depth, or true skin/nail anatomy. Palm filling is heuristic, particularly when the hand rotates or fingers overlap. Small exposed portions of the original colored lines are inpainted in the refined version. Illustrative details can vary between frames. The source recording and website have not been replaced.

Reproduce from the project root:

```sh
/opt/anaconda3/envs/umetrack/bin/python scripts/render_surface_refined.py
```
