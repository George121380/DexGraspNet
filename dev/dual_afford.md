### DualAfford 项目中的点云采样机制

以下基于 `third_party/DualAfford/` 代码的快速研读与交叉检索，归纳该项目点云采样的来源、下采样策略与关键实现位置。

### 核心结论

- 点云来源：
  - 从渲染相机的深度结果构建 3D 点云。通过 `Camera.compute_XYZA_matrix` 把深度映射为 `XYZA`，并用 `A>0.5` 作为有效像素的掩码得到原始点集。

- 初步子采样（随机）：
  - 在多个入口（例如 `data.SAPIENVisionDataset.__getitem__`、`utils.get_part_pc`、`collect_data_SAC.get_predict_aff_map`）中，对有效点先做随机打散，并扩展/截断到约 3 万点规模（不足则重复补齐，超出则截断）。
  - 常见代码形态：先 `np.random.shuffle(idx)`，若不足则重复拼接，最后 `idx = idx[:30000-1]` 并选择对应点。
  - 有一处坐标平移：`pc[:, 0] -= 5`（X 方向减 5），用于与相机/可视化工具坐标系对齐。

- 关键下采样（FPS）：
  - 训练与推理阶段普遍采用“最远点采样（Furthest Point Sampling，FPS）”将点数规整到配置值（默认 `num_point_per_shape=8192`）。
  - 调用点示例：
    - `train_actor.py`、`train_affordance.py`、`train_critic.py`、`train_collaborative_adaptation.py` 中：
      - 将 batch 内每个样本的 `part_pc` 拼成 `pcs`（形如 `B x N x 3`），再执行：
        - `idx = furthest_point_sample(pcs, conf.num_point_per_shape)`
        - `pcs = pcs[input_pcid1, idx.reshape(-1), :]` → 整理成 `B x num_point_per_shape x 3`
  - RL 数据采集里（`collect_data_SAC.get_predict_aff_map`）：
    - 对单帧点云执行 `furthest_point_sample(pc, 10000)`，得到 `N=10000` 的子集。

- FPS 实现位置：
  - `code/models/pointnet2_ops/pointnet2_utils.py` 中定义 `furthest_point_sample = FurthestPointSampling.apply`，其 `forward` 调用 C++/CUDA 扩展 `_ext.furthest_point_sampling`。
  - 若 `_ext` 扩展无法直接导入，将尝试基于 `torch.utils.cpp_extension.load` 在本地 JIT 编译（需 CUDA 环境）。

- PointNet++ 模块内部采样：
  - `code/models/pointnet2_ops/pointnet2_modules.py` 的 Set Abstraction（SA/MSG）层也使用 `furthest_point_sample` 选取中心点，然后通过球查询（`ball_query`）与分组（`QueryAndGroup`）聚合局部邻域特征。

### 数据流与调用链（简述）

1) 数据加载（以 `data.SAPIENVisionDataset.__getitem__` 为例）：
   - 读取 `cam_XYZA_*.h5`，`Camera.compute_XYZA_matrix` 生成 `H x W x 4 (XYZA)`。
   - 取 `A>0.5` 的有效 3D 点，随机打散，补齐/截断到约 3 万点，执行坐标系转换（`world` 或 `cambase`）。
   - 返回 `1 x N x 3` 的张量作为 `part_pc`。

2) 训练脚本（以 `train_actor.py`/`train_affordance.py`/`train_critic.py` 为例）：
   - 将 batch 的 `part_pc` 沿 batch 维拼接为 `pcs: B x N x 3`（或 `torch.cat([...], dim=0)`）。
   - 使用 `furthest_point_sample(pcs, conf.num_point_per_shape)` 得到每个样本的 `npoint` 索引，并聚合出最终的 `B x npoint x 3`。

3) RL 采集（以 `collect_data_SAC.get_predict_aff_map` 为例）：
   - 从相机 `XYZA` 直接构建当前帧的点云，先随机扩展到 3 万点，再用 `furthest_point_sample(pc, 10000)` 得到 `N=10000` 的子点云用于后续评分与可视化。

### 关键文件与片段（路径）

- 生成/预处理点云：
  - `code/data.py` → `SAPIENVisionDataset.__getitem__` 的 `part_pc` 分支
  - `code/utils.py` → `get_part_pc`、`render_pts_label_png`（仅可视化导出）
  - `code/collect_data_SAC.py` → `get_predict_aff_map`

- 最远点采样（FPS）：
  - `code/models/pointnet2_ops/pointnet2_utils.py` → `furthest_point_sample`
  - `code/models/pointnet2_ops/pointnet2_modules.py` → SA/MSG 层内部中心点选取与邻域分组

- 采样调用处（选）：
  - `code/train_actor.py`、`code/train_affordance.py`、`code/train_critic.py`、`code/train_collaborative_adaptation.py`

### 重要超参数

- `--num_point_per_shape`（默认 8192）：训练阶段每个样本采样点的目标数量。
- `collect_data_SAC.get_predict_aff_map`：FPS 采样数量固定为 `10000`。
- 初步随机子采样规模：多处使用 `≈30000` 作为上限/目标规模。

### 工程注意点

- `pointnet2_ops` 为自定义 C++/CUDA 扩展：
  - 如果无法直接导入，会尝试 JIT 编译；需要可用的 CUDA 编译环境，且编译耗时较长。
  - 仅 CPU 环境下无法使用该 FPS 扩展（默认 `with_cuda=True`）。如需 CPU 回退，需自行替换为纯 PyTorch/Faiss/PyTorch3D 的 FPS 实现。

- 坐标平移 `pc[:, 0] -= 5`：
  - 该平移在多处出现，和相机位姿/离线渲染工具（例如 `utils.render_pts_label_png` 调用的 Thea）使用的可视化坐标系有关。

### 小结

- DualAfford 的点云由相机深度重建得到，经随机均匀子采样（~3 万点）与坐标处理后，统一使用最远点采样（FPS）规整到任务所需的点数（训练默认 8192，采集/可视化可为 10000）。
- FPS 的实现依赖项目内置的 `pointnet2_ops` CUDA 扩展，Set Abstraction 模块内部也依赖该采样机制做层级特征提取。


### Affordance 模型训练机制（深入）

- 模型定义（第一阶段，单接触点）：`code/models/model_aff_fir.py`
  - Backbone：PointNet++ SSG（自定义的 `PointNet2SemSegSSG`），包含多层 SA（npoint: 1024/256/64/16，半径 0.1/0.2/0.4/0.8）和 FP 金字塔，最后经 1×1 卷积得到全局特征 `feat_dim`。
  - 特征输入：
    - 任务特征 `task`：`mlp_task(task_input_dim -> task_feat_dim)`，task_input_dim 由动作类型决定（推/倒/抓取=3，旋转=1）。
    - 接触点特征 `cp1`：`mlp_cp(3 -> cp_feat_dim)`。
    - 全局点云特征：PointNet++ 输出的每样本全局描述（使用 `whole_feats[:,:,0]`）。
  - 打分头 `ActionScore`：`concat([global_feats, task_feats, cp1_feats]) -> 128 -> 1`，输出二分类 logit；经 `sigmoid` 得到 `pred_score ∈ (0,1)`。
  - 前向：`Network.forward(pcs, task, cp1)`，并提供 `inference_whole_pc(pcs, task)` 逐点推理整云得分（把每个点作为cp1）。
  - 损失：`L1Loss(pred_score, gt_score)`（逐样本平均）。

- 训练数据管线：`code/train_affordance.py`
  - 数据集：`SAPIENVisionDataset` 返回 `part_pc`（相机可见点云，见前文）、`task`、`ctpt1/ctpt2`、`dir1/dir2`、`success` 等。
  - 点云预处理：
    - 先拼 batch：`pcs = torch.cat(part_pc)`；
    - 使用 `furthest_point_sample(pcs, num_point_per_shape)` 将每样本规整到 `num_point_per_shape`（默认 8192）。
  - 标注构造（teacher来自 Actor/Critic）：
    - 第一阶段（`model_aff_fir`）：
      - 使用已训练好的 `actor` 生成若干候选方向 `recon_dir1`（rvs=conf.rv_cnt），并用 `critic` 对这些候选打分，取 top-k 平均得到 `gt_score`；
      - 使用 `affordance` 的 `forward(pcs, task, ctpt1)` 预测 `pred_score`，监督信号为 `L1(pred, gt)`；
    - 第二阶段（`model_aff_sec`，类比处理 ctpt2 与 dir1）：结构与损失形式相近，只是输入包含 ctpt2、dir1，用于双接触点场景。
  - 优化与调度：Adam；StepLR（`step_size` 与 `gamma` 可配置）。
  - 记录与可视化：TensorBoard（`tb/train`、`tb/val`），周期性保存 checkpoint 与验证。

- 关键超参（摘自 `train_affordance.py`）：
  - `num_point_per_shape=8192`，`feat_dim=128`，`task_feat_dim=32`，`cp_feat_dim=32`，`dir_feat_dim=32`；
  - `rv_cnt`（Actor采样数，默认100），`topk`（取前K均值作为GT），`lr=1e-3`，`weight_decay=1e-5`，`lr_decay_every=5000`，`lr_decay_by=0.9`；
  - `coordinate_system` 支持 world/cambase；采样/对齐策略见前文点云部分。

- 训练依赖：
  - 需要预先训练好的 `actor` 与 `critic`（路径与 epoch 在 conf 中指定），用于为affordance提供 teacher 的 `gt_score`。
  - 需要 `pointnet2_ops` CUDA 扩展以获得高效 FPS、ball query、grouping、three_nn/interpolate 等操作。


### 接触点选择机制（cp 的来源与用法）

- 采集阶段（collect_data_SAC.py）：
  - 先用相机获得目标链接的像素掩码 `gt_all_link_mask`；若启用 PAM（Where2Act）预估，则用 `pred_aff_score` 的高分区生成 `select_seg_mask`，否则直接用 `gt_all_link_mask`。
  - 在 `select_seg_mask` 内随机采样两个像素 (x1,y1),(x2,y2)，要求二者不同且在世界坐标下的距离 > 0.25；
  - 通过 `get_contact_point(cam, cam_XYZA, x, y)` 将像素反投影为世界坐标 `position_world`，记为 `contact_point_world1/2`，并写入采集的 JSON 结果。

- 数据集阶段（data.py → SAPIENVisionDataset）：
  - `load_data` 从采集 JSON 读取 `position_world1/2` 并作为 `ctpt1/ctpt2`；可选 `exchange_ctpts=True` 进行 cp 交换增强（把1、2对调再入库）。
  - `__getitem__` 返回 `ctpt1/ctpt2`（以及 `task/dir` 等），供训练脚本使用。

- 训练阶段（train_*）：
  - 第一阶段 `model_aff_fir`：
    - 使用数据集中给定的 `ctpt1` 作为接触点输入；
    - 需要对整云打分时，`inference_whole_pc` 将点云中的每个点当作“候选 cp1”，逐点预测得分图。
  - 第二阶段 `model_aff_sec`：
    - 固定 `cp1`，将点云各点视为候选 `cp2`，逐点预测“配对接触点”的得分图。
  - 在 `train_critic.py` 等脚本中，会用 `affordance.inference_whole_pc` 对整云得分，取 `topk` 最高分的若干接触点作为候选（如 `rvs_ctpt` 个），再交给 `actor/critic` 做后续评估与监督信号构造。

### Affordance 打分模型的 Ground Truth Label 构造（Teacher 蒸馏）

- 总体思想：不直接用环境仿真结果作为标签，而是用“已训练的 Actor + Critic”作为 teacher 生成软标签。对每个样本，先由 Actor 在接触上下文中采样多个方向，交给 Critic 评分，再把这些分数取 top-k 的平均作为当前样本的 ground truth（连续分数 ∈ [0,1]）。训练时用 L1Loss 让 Affordance 预测逼近该软标签。

- 第一阶段（`model_aff_fir`，单接触点）：参见 `code/train_affordance.py` 中 `forward()`
  1) 预测：`pred_score = network.forward(pcs, task, ctpt1)`
  2) Teacher 采样与打分（no_grad）：
     - 用 Actor 采样方向（6D）`recon_dir1 = actor.actor_sample_n(pcs, task, ctpt1, rvs=rv_cnt)`
     - 用 Critic 评估 `gt_scores = critic.forward_n(pcs, task, ctpt1, recon_dir1, rvs=rv_cnt)`，得到 `B×rv_cnt` 分数
     - 聚合标签：`gt_score = topk(gt_scores, k=topk).mean(dim=1)`（B×1）
  3) 损失：`L1Loss(pred_score, gt_score)`

- 第二阶段（`model_aff_sec`，双接触点）：参见 `code/train_affordance.py` 中 `forward()`
  1) 预测：`pred_score = network.forward(pcs, task, ctpt1, ctpt2, dir1)`
  2) Teacher 采样与打分（no_grad）：
     - 固定 `ctpt1/ctpt2/dir1`，Actor 采样第二个方向 `recon_dir2 = actor.actor_sample_n(pcs, task, ctpt1, ctpt2, dir1, rvs=rv_cnt)`
     - Critic 评分 `gt_scores = critic.forward_n(pcs, task, ctpt1, ctpt2, dir1, recon_dir2, rvs=rv_cnt)`
     - 聚合标签：`gt_score = topk(gt_scores, k=topk).mean(dim=1)`
  3) 损失：同上用 `L1Loss`

- 关键超参：
  - `rv_cnt`（Actor 采样数，默认 100）：每个样本的候选方向数量
  - `topk`（聚合阈值，默认 100）：取每样本 top-k 的均值作为软标签；可调节“只看最好候选”的程度
  - 标签性质：连续软标签（非二值），范围在 [0,1]



