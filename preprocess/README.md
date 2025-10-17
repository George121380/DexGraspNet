## 预处理模块说明（preprocess）

该目录包含用于对象点云采样、手部姿态恢复、抓取中心计算、可视化与二阶段（affordance）生成的脚本与工具。

### 文件概览

- `__init__.py`
  - 将 `preprocess` 设为 Python 包。

- `utils.py`
  - 通用工具与数据装载：
    - 构建手部姿态向量 `build_hand_pose_tensor`
    - 计算抓取中心点 `get_grasp_center_point`
    - 物体点云采样（全表面/可见外表面/体素近似外壳）
    - 简单的 FPS（Farthest Point Sampling）和补点工具
    - 一站式装载 `load_models_and_data`（返回物体点云、手部表面点、模型实例等）

- `preprocess.py`
  - 针对单个对象的单一姿态进行预处理：
    - 采样物体点云，计算左右手代表性抓取中心点
    - 输出至 `preprocess/results/<object_name>/`（`obj_points.npy` 与汇总版 `grasp_pairs.npy`）

- `preprocess_object_all_poses.py`
  - 针对单个对象的所有姿态进行预处理与聚合：
    - 复用手模型与物体点云，仅逐姿态更新参数
    - 持续写入/更新汇总 `grasp_pairs.npy`（每姿态的左右手 `qpos` 与抓取中心）

- `visualize_preprocessed.py`
  - 可视化预处理后的结果：
    - 显示物体点云、左右手抓取中心与手部网格
    - 输入 `preprocess/results/<object_name>/` 与姿态索引，输出交互式 HTML

- `visualize_affordance.py`
  - 交互式三联图可视化左右手的距离型亲和值（affordance）：
    - 支持外表面/可见外表面采样策略
    - 分别在两个子图中渲染右手与左手的点云亲和值热力

- `aff_sec.py`
  - 二阶段（second-stage）右手 affordance 原型（高斯场）与可视化（单对象单姿态）：
    - 从 `preprocess/results/<object_name>/` 读取 `obj_points.npy` 与 `grasp_pairs.npy`
    - 以右手抓取中心及（可选）指尖网格点为高斯中心，计算物体点云上的场并着色
    - 生成交互式 HTML

- `aff_sec_batch.py`
  - 批处理版本的二阶段（second-stage）右手 affordance 生成：
    - 遍历输入根目录下各对象及其所有姿态
    - 为每个姿态保存：左手关键点（可选为抓取中心）与右手的高斯场打分（归一化）
    - 输出至 `preprocess/aff_sec_result/<object_name>/aff_sec_pairs.npy`

- `aff_first.py`
  - 提取第一只手（默认 left）的 pairs：
    - 从 `grasp_pairs.npy` 生成仅该手的子集，包含 `qpos`、中心点与（可选）从手部网格采样的关键点
    - 支持可选重算中心点（基于手部表面点与物体点云）
    - 输出文件：`aff_first_pairs_<hand>.npy`

### 典型输入/输出（简要）
- 输入：
  - `preprocess/results/<object_name>/obj_points.npy`
  - `preprocess/results/<object_name>/grasp_pairs.npy`（聚合版，含各姿态 `qpos` 与中心点）
- 输出：
  - 可视化 HTML（在对应目录或自定义路径）
  - 批处理二阶段结果 `preprocess/aff_sec_result/<object_name>/aff_sec_pairs.npy`

### 备注
- 若使用 GPU，可将脚本参数 `--device` 设为 `cuda` 或使用 `auto` 自动检测。
- 可视化依赖 Plotly，生成的 HTML 可在浏览器中直接打开。
