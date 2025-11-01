整体目标：
我的整个pipeline需要实现输入一个物体名称，最终输出两只shadowhand灵巧手的抓取pose

详细设计：
1. 针对输入的物体名称，首先需要从pipeline_input文件夹（/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline/pipeline_input）中读取对应名称物体的点云文件（obj_points.npy），这个点云输入给后面affordance model。
2. 我已经训练好了两个affordance model，第一个affordance model（相关内容见：/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_first）以点云为输入，输出一个affordance map（点云中每个点的affordance 分数）随后根据分数在这些点云中采样出一个keypoint。（inference方法你可以参考：/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_first/evaluate.py）
3.第二个affordance model（相关代码见：/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second）以点云和一个keypoint作为输入，输出第二个affordance map，随后你再根据第二个affordance map采样出第二个keypoint（inference 方法你可以参考：/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/train_eval_second/evaluate.py）
4.我训练好了一个pose generator（相关代码放在：/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet2），这个generator以点云和两个keypoints作为输入，输出两只shadowhand的grasping pose（inference 代码可以参考/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/DexGraspNet2/src/eval/export_bimanual.py）
5.我有一个pose optimization程序（相关代码见：/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/third_party/BimanGrasp-Generation），我想要用4中输出的pose来初始化optimizer，然后进行100步的优化得到最终的pose。


环境要求：
步骤2，3需要在名为pn的conda环境下运行。步骤4需要在名为DexGrasp的conda环境下运行。步骤5使用bimangrasp这个conda环境运行

实现要求：
1.我希望代码简单干净易读，针对每个步骤可以写一个infer代码放在utils文件夹中（/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline/utils）。
2.pipeline.py是我使用时运行的代码（/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline/pipeline.py）我希望运行这个代码后首先load好两个affordance model（这两个load比较慢），随后一个server可以持续接受物体名称输入，每次输入后处理这个物体，运行完整pipeline
3.实现过程中我希望你可以分步调试好每个模块再整体测试来提升测试效率
4.每个模块运行后都要进行可视化，如2，3需要记录网页来展示3d点云的value以及sample的点（相关代码中包含了可视化方法你可以参考），4，5也要记录网页来可视化
5.可视化内容，中间结果等都可以记录在这个文件夹中/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline/outputs，每次实验要做好记录
6.当前我提供给你的模型是我实现方法的模型，但之后我可能会替换其中一些部分作为我的baseline实验，所以我需要你确保你的代码是便于进行模块替换更改的。
7.使用yaml config来管理相关参数，参数放置在config文件夹下（/media/george/Projects/Research/2026-CVPR-BiDexHand/affordance-bidex/pipeline/configs）
8.运行过程中需要包含丰富且清晰的print/log，实时说明当前进度、正在执行的步骤（如“预加载模型”“第一阶段推断”“DexGrasp推断”“优化中step x/100”等）。