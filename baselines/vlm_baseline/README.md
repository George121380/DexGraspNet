VLM Affordance Baseline (two points: left/right)

Overview
- This baseline queries a VLM (mocked by default) with up to 4 images per object and returns exactly two grasp contact points on the object: left and right.
- Optional 3D backprojection uses rendered depth and camera parameters to map the 2D points into 3D.

Setup
1) Python 3.10+ recommended
2) Install dependencies:
```
pip install -r requirements.txt
```

Config
- See `configs/vlm_baseline.yaml` for defaults.

Pipeline (example)
1) Render multi-view images with intrinsics/extrinsics:
```
python scripts/render_objects.py \
  --dataset-dir third_party/BimanGrasp-Dataset/Object-Release-v1 \
  --out-dir data/vlm_renders \
  --views 12 --width 640 --height 480
```

2) Query VLM (mock) for two points per object (max 4 images per object):
```
python scripts/query_vlm.py \
  --renders data/vlm_renders \
  --model gpt-5-vision \
  --max-images 4 \
  --two-points-only \
  --outputs-root outputs/vlm_baseline
```

3) Backproject to 3D:
```
python scripts/fuse_and_backproject.py \
  --renders data/vlm_renders \
  --vlm-json-root outputs/vlm_baseline
```

4) Evaluate and visualize:
```
python scripts/evaluate_vlm_grasps.py \
  --fused-root outputs/vlm_baseline
```

Notes
- Rendering requires an offscreen OpenGL context. If `pyrender` fails, ensure a proper EGL/OSMesa setup or consider running in a desktop session.
- If a mesh is not found for an object, the renderer will skip it.


