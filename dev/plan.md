### 目标

为每条 (object, pose) 数据计算物体表面点的affordance，并进行并排可视化：
- 左侧：原始mesh + 左右手抓取姿态（复用 `visualize_dataset.py` 的逻辑和数据加载）。
- 右侧：在物体表面采样 k=1024 个点，计算点到双手模型任意最近点的距离，并将距离映射到 [0,1] 作为affordance，使用颜色可视化点云。

### 数据与模型
- 物体：`third_party/BimanGrasp-Dataset/Object-Release-v1/<object_name>/coacd/decomposed.obj`
- 物体采样：`ObjectModel.initialize(object_code)` 已提供 `surface_points_tensor`（此处默认采用mesh顶点采样再随机下采样/重复以达到k）。
- 手模型：`HandModel` 已支持 `set_parameters` 并能通过 `get_surface_points()` 输出手在世界坐标下的表面点集合（我们将设置 `n_surface_points`，或直接使用构造中的默认顶点采样逻辑）。

### Affordance 计算
1. 载入指定 `object_name` 与 `num`（pose索引），读取 `BimanGrasp-Dataset-Release-v1/<object_name>.npy` 的第 `num` 项。
2. 通过 `HandModel` 构建左右手实例，调用 `set_parameters` 应用抓取姿态，得到两手在世界坐标的表面点云 `P_hand`（合并左右手）。
3. 通过 `ObjectModel.initialize(object_name)` 并设置 `object_scale_tensor`，获取物体表面点云 `P_obj`（大小约等于k，必要时随机下采样或重复补齐到k=1024）。
4. 计算每个 `p ∈ P_obj` 到集合 `P_hand` 的最近邻距离：`d(p) = min_{q∈P_hand} ||p - q||₂`。
5. 将距离映射为affordance：`a(p) = 1 - clip(d(p)/d_max, 0, 1)`，其中 `d_max` 可选：
   - 常数阈值（如 0.05m），或
   - 当前样本中的分位数（如 95%分位）。
   默认采用常数阈值（可通过CLI参数调整）。

### 可视化
- 采用 Plotly：
  - 左侧：直接复用 `visualize_dataset.py` 的三方对象生成（左右手mesh + 物体mesh）。
  - 右侧：使用 `Scatter3d` 绘制 `P_obj`，依据 `a(p)` 渐变上色（如 `Viridis`）。
  - 两个子图使用 `make_subplots(rows=1, cols=2, specs=[[{"type":"scene"},{"type":"scene"}]] )`。

### CLI 设计
`python visualize_affordance.py --object_name <object_name> --num <num> --k 1024 --dmax 0.05 --result_path third_party/BimanGrasp-Dataset/BimanGrasp-Dataset-Release-v1`
- `--object_name`：物体编码（与 `.npy` 文件名一致）。
- `--num`：使用的pose索引。
- `--k`：物体表面点采样数量，默认 1024。
- `--dmax`：距离归一化上限，默认 0.05。
- `--result_path`：抓取结果 `.npy` 的目录。

### 实现要点
- 重用 `visualize_dataset.py` 中的数据解析逻辑来构造 `hand_pose`。
- 使用 `HandModel.get_surface_points()` 直接得到手在世界坐标下的点。
- 使用 `ObjectModel.initialize()` 后的 `surface_points_tensor[0]` 作为物体点云基础，并根据 `k` 随机下采样或重复补齐。
- 最近邻距离计算：为避免过慢，采用 `torch.cdist`（在CPU上也可运行）。
- 颜色映射：采用 `colorscale='Viridis'`，`marker=dict(color=affordance, colorscale='Viridis', cmin=0, cmax=1)`。
- 清理临时变量，不写入临时文件。

### 交付物
1. 计划文档：`dev/plan.md`（本文件）。
2. 运行脚本：`third_party/BimanGrasp-Dataset/visualize_affordance.py`。
3. 使用示例：
   ```bash
   python third_party/BimanGrasp-Dataset/visualize_affordance.py \
     --object_name Asus_M5A99FX_PRO_R20_Motherboard_ATX_Socket_AM3 \
     --num 0 --k 1024 --dmax 0.05 \
     --result_path third_party/BimanGrasp-Dataset/BimanGrasp-Dataset-Release-v1
   ```



