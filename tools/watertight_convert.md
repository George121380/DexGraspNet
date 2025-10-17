## watertight_convert 使用说明

单命令将 PartNet-Mobility/SAPIEN 物体（URDF + 网格）转换为 watertight 网格，封装并调用现有工具：
- `tools/urdf_to_watertight.py`
- `tools/simplify_collision.py`
- `tools/check_and_render_mesh.py`

支持模式：合并（merged）、逐 link（per-link）与自动（auto，失败时自动回退为“合并 OBJ + 体素重建”）。

### 环境准备
- 建议在 urdf2mesh 环境中运行：
```bash
source ~/.bashrc  # 如有
conda activate urdf2mesh  # 或：source .venv-urdf2mesh/bin/activate
```

### 快速上手
- 自动模式（必要时自动回退），同时输出 30 万面简化与预览：
```bash
python tools/watertight_convert.py \
  --obj-dir data/sapien_data/partnet-mobility-dataset/100015 \
  --mode auto \
  --target-faces 300000 --watertight-target
```

- 仅合并所有 link（URDF 路径成功时）：
```bash
python tools/watertight_convert.py --obj-dir <OBJ_DIR> --mode merged
```

- 保留 link 结构并更新 URDF（URDF 路径成功时）：
```bash
python tools/watertight_convert.py --obj-dir <OBJ_DIR> --mode per-link
```

- 强制走回退（合并 OBJ + 体素重建）：
```bash
python tools/watertight_convert.py --obj-dir <OBJ_DIR> --force-fallback
```

### 常用参数
- `--obj-dir`: 包含 `mobility.urdf` 与 `textured_objs` 的目录（必需）
- `--out-dir`: 输出目录，默认 `<obj-dir>/watertight`
- `--mode`: `merged` | `per-link` | `auto`
- `--voxel-size`:（URDF 路径）体素重建尺寸（米），越小细节越多但更耗内存
- `--simplify-ratio`:（URDF 路径）几何修复后的简化比例
- `--min-component-tris`: 移除小连通分量的最小面数阈值
- `--target-faces`: >0 时将最终网格简化到目标面数
- `--watertight-target`: 简化时尽量保持 watertight（使用 `simplify_collision.py`）
- `--no-preview`: 跳过预览图渲染
- `--force-fallback`: 跳过 URDF 工具，直接走“合并 OBJ + 体素重建”
- `--fallback-resolution`: 回退体素重建的分辨率（bbox 最大维的划分数，默认 64）
- `--preview-size`: 预览分辨率（宽 高），默认 1024 768

### 输出
- 合并网格：`watertight/merged_collision.obj` 或 `merged_collision_<faces>_wt.obj`
- 预览图：`watertight/preview.png`
- QA（简化时）：`watertight/qa_simplified_<faces>.json`
- 回退路径附加：`watertight/qa_wrapper.json`

注意：回退路径不会更新 URDF；仅当选择 `--mode per-link` 且 URDF 路径成功时，才会由 `tools/urdf_to_watertight.py` 写出 `mobility_watertight.urdf`。

### OOM/内存建议
- 推荐使用 `--mode auto`；若 OOM，脚本会自动回退
- 可调大 `--voxel-size`（URDF 路径）或减小 `--simplify-ratio`
- 回退路径可将 `--fallback-resolution` 调小（如 48/32）以降内存，但细节会更粗
- 脚本默认限制线程数以降低内存峰值

### 批处理示例
```bash
source .venv-urdf2mesh/bin/activate
ROOT=data/sapien_data/partnet-mobility-dataset
for ID in 100015 100017 100020; do
  python tools/watertight_convert.py \
    --obj-dir ${ROOT}/${ID} \
    --mode auto \
    --target-faces 300000 --watertight-target
done
```

### 故障排查
- PyVista 预览相关警告（xvfb/OSMesa）：仅用于离线截图，可忽略
- 未生成结果：检查 `textured_objs` 是否包含 OBJ；查看终端错误信息
- 需要更严格 watertight：开启 `--watertight-target` 并尝试更高 `--target-faces`



