一、创建环境并安装

建议新建一个干净环境（示例用 conda，也可用 venv）

conda create -n o3dml python=3.12 -y
conda activate o3dml


安装 PyTorch（CUDA 运行时随轮子一起带，不要求你本机装同版 CUDA）

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


PyTorch 官方建议用专属 index-url 安装带 CUDA 的轮子。
PyTorch Forums

安装 Open3D（0.19 起正式支持 Python 3.12 / 适配新栈）

pip install open3d


Open3D v0.19 宣布支持 Python 3.12 与新依赖栈；Open3D-ML 模块已集成在 open3d 包内，无需单独装 open3d-ml。
open3d.org
GitHub

快速自检（能 import 就说明安装 OK）

python - <<'PY'
import open3d as o3d
import open3d.ml.torch as ml3d
import torch
print("Open3D:", o3d.__version__)
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
PY


Open3D-ML 文档推荐用 import open3d.ml.torch as ml3d 测试。
GitHub

二、最小可用：加载预训练网络跑一次（验证管线）

先用自带管线跑通一次语义分割（RandLA-Net），确认依赖、GPU、数据流水线均工作正常（示例使用文档里的流程，可替换为你自己的点云/数据集）。

import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "ml3d/configs/randlanet_semantickitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.RandLANet(**cfg.model)
dataset = ml3d.datasets.SemanticKITTI(dataset_path="/path/to/SemanticKITTI", **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

ckpt_path = "./logs/randlanet_semantickitti_202201071330utc.pth"
os.makedirs("./logs", exist_ok=True)
if not os.path.exists(ckpt_path):
    os.system(f"wget https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth -O {ckpt_path}")

pipeline.load_ckpt(ckpt_path=ckpt_path)
split = dataset.get_split("test")
data = split.get_data(0)
out = pipeline.run_inference(data)  # {'predict_labels', 'predict_scores'}
print("OK:", {k: (v.shape if hasattr(v,'shape') else type(v)) for k,v in out.items()})


以上流程与路径配置取自 Open3D-ML README 的示例（加载 config→构建模型/数据集→下载权重→推理/评测）。
GitHub

三、把网络当“PointNet 风格”的特征提取器来用

你要的是“像 PointNet 一样”的点云全局特征（例如一个 1024-D 向量），常见做法有两条：

方案 A（更简单）：用官方的 Open3D-PointNet 示例仓库

这个仓库是 Open3D 生态内的 PointNet（PyTorch）实现，演示如何用 Open3D 读点云并跑 PointNet，可直接改成“拿倒数第二层特征”。

快速步骤：clone → 安装依赖 → 跑 open3d_pointnet_inference.ipynb，或调用其中的 pointnet.py 并把分类头前那一层的特征拿出来做 embedding。

仓库与官方文档“Example projects”里都给出了入口。
GitHub
open3d.org

方案 B（无需额外仓库）：用 Open3D-ML 自带模型取特征

选一个点式架构（如 RandLA-Net / PointTransformer / PVCNN 等），前向时拿“分类/分割头之前”的特征图，然后做全局池化（max/avg）得到全局 embedding。

Open3D-ML 文档列出了可用模型类（ml3d.models.RandLANet / PointTransformer / PVCNN …），它们都是 PyTorch 模型，拿中间层很方便。

模型类与用法参见 API 文档（如 open3d.ml.torch.models.*）。
open3d.org
+2
open3d.org
+2

小提示：PVCNN 在结构上“点+体素”的混合，底层仍有 PointNet 风格分支；做特征抽取时取其点分支的聚合特征效果也不错。
open3d.org

四、常见坑与规避

Python / Open3D 版本：请确保 open3d>=0.19（支持 Python 3.12）。如果你装的是旧版，会在 3.12 上报不兼容。
open3d.org

TensorFlow 冲突：从 v0.18 起，Linux 上的 PyPI 轮子与 TF 的原生算子打包存在冲突；若你只用 PyTorch（我们就是），无需理会。
GitHub

PyTorch CUDA 轮子：用官方 --index-url 安装带 CUDA 的版本，不必与系统 CUDA 完全一致，但显卡驱动需足够新（A6000 一般没问题）。
PyTorch Forums

五、你可能会用到的官方资料

Open3D 0.19 发布说明（确认 Python 3.12 支持）。
open3d.org

Open3D-ML 总览/安装与示例代码（open3d.ml.torch 入口、加载 config、预训练推理）。
GitHub
open3d.org

Open3D 文档中的示例工程（Open3D-PointNet / PointNet++）。
open3d.org

Open3D-PointNet 仓库（直接上 PointNet，便于抽特征）。
GitHub

如果你把“输入点云的格式/尺寸 & 需要的向量维度（比如 256/512/1024）”告诉我，我可以直接给你一段最小的特征提取脚本（基于上述两条方案之一，去掉分类头、输出全局 embedding），并按你的数据接口替换读取部分。