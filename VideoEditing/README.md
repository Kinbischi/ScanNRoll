# VideoEditing — zoom insert for the presentation video (side project)

**Separate side thing, not part of the analysis pipeline.** Nothing in the pipeline imports this folder; it only
*reads* the processed cache (via `dataAnalysisSetup.load()`) and the plotting helpers, exactly like the first
`dataAnalysis.py` cell with `PATH_LAYOUT = "linear"`.

It takes the existing video `transitionToScanSequence_smooth.mp4` (real printing footage → transition → the
dataAnalysis 3D view; 1920x1080, 30 fps, 800 frames) and produces two versions with a **zoom into the point cloud**
inserted at 0:15:

| File | Content |
|---|---|
| `transitionToScanSequence_smooth_zoom.mp4` | A: zoom in → drift + orbit → zoom out (34.3 s) |
| `transitionToScanSequence_smooth_zoom_profileLine.mp4` | B: same camera move + a magenta line (#F20782, the laser colour in the footage) tracing one scan profile, then a stack of 12 profiles receding along the belt |

All videos (`*.mp4`) and the generated frames/caches (`work/`) are git-ignored; only the code is tracked.

## Timeline of the new video

| New time | What |
|---|---|
| 0:00–15.00 | original frames 0–449 (unchanged) |
| 15.00–22.63 | **zoom insert** (229 frames): 8-frame dissolve in from the original close-up; ease in and level out looking along the bead (profiles read as arcs of points) → slow drift along the belt → orbit to a side view (profiles read as a comb of point columns) → pull back to the exact close-up. Version B draws the profile line at 17.3–18.1 s, steps through 11 more profiles (every 4th, 1.8 mm apart) and fades out by 20.1 s |
| 22.63–34.30 | **re-rendered tail** (original frames 450–799, static close-up) with smoother effects at the original event times (+7.63 s): blue edge points 24.70 s (crisp dots sweeping head→tail; hidden far-edge points semi-transparent), fade out 26.47 s; body colour sweep 28.87 s; rupture-tip sweep 30.23 s |

## How to run (from the repo root, Python 3.11)

```bash
py -3.11 VideoEditing/zoomInsert/render_insert.py      # insert base frames -> work/frames_base (~40 s)
py -3.11 VideoEditing/zoomInsert/tail_effects.py       # tail colour states + depth -> work/tail_cache.npz (~30 s)
py -3.11 VideoEditing/zoomInsert/compose.py A          # splice + encode version A (~2 min)
py -3.11 VideoEditing/zoomInsert/compose.py B          # version B (~5 min, overlay drawn per frame)
```

Needs `pyvista`, `opencv-python`, `scipy` and an **ffmpeg with libx264 + aac** (`FFMPEG` env var, else `ffmpeg` on
PATH, else the Anaconda package-cache copy). Encode: H.264 CRF 16, yuv420p, 30 fps, the original's silent stereo
audio padded to the new length; untagged colour like the original (ffmpeg default matrices both ways).

## Files (`zoomInsert/`)

| File | Role |
|---|---|
| `render_lib.py` | Paths, data load, linear-path layout, viewer-identical layer extraction (`layer_points`, same voxel picks), `segment_table()` |
| `insert_renderer.py` | `InsertScene`: off-screen PyVista scene in video space (2x supersampling optional) with the fitted camera + data scaling; projection helper for 2D overlays |
| `path_lib.py` | Orbit camera parametrisation (target, azimuth, elevation, log-distance, roll) + C1 Hermite path |
| `make_path.py` | Camera keyframes/times of the insert (edit here to change the move) |
| `render_insert.py` | Renders the 229 insert base frames (shared by A and B) |
| `profile_overlay.py` | Version-B profile line: timeline, stride, trail, colours |
| `tail_effects.py` | Re-rendered static tail + effect timeline/style (edge points, body / rupture sweeps) |
| `compose.py` | Splices original + insert + tail and encodes |
| `pose_scaled_zscale.npy`, `orbit_close.npy` | The reconstructed close-up camera (see below) |

## How the close-up camera was reconstructed

The 3D part of the existing video was rendered by another session whose camera pose was not available, so it was
fitted to the frames (static close-up, original frames 412–511):

1. **Which segment:** the strip edges fix the view up to one angle; segment length/gap only pin their ratio, so the
   phase colouring of frame 678 (short head, long blue body, visible green shoulder, red tip; next segment green
   → blue) identified **segment 49** (z ≈ 17.2 m on the linear path).
2. **Pose:** grid over focal length × angle on label masks, then 7-DOF refinement on full-resolution VTK renders.
3. **Geometry mismatch found:** the best fit needs the data drawn **~23 % taller (s_y = 1.233) and ~1.8 % longer
   along the belt (s_z = 1.018)** than the processed data at true scale, with a 31.9° vertical view angle (the
   handover's 26.7° did not reproduce the frame). With that, render vs. video differs in 0.8 % of pixels (edges only).
   The insert inherits this scaling so the splice is seamless — i.e. the presentation video's 3D view shows the
   heights exaggerated by ~23 %.

The exploratory fitting scripts were not kept; the result is stored in the two `.npy` files.

## Caveats

- At the left frame edge the fitted camera shows a ~17x50 px sliver of the previous segment's floor gap that the
  original close-up does not; it is painted with the floor colour in the static tail and faded in the first/last
  14 insert frames (`tail_effects.fill_notch`) — cosmetic only.
- Everything after 0:15 in the new video is 7.63 s later than in the original (matters for music sync).
