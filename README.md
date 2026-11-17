# BioCross 推理部署分支

`hwd_deploy` 仅保留单样本推理；训练和下游分类代码见 `main`。

## 最小模块

```text
infer_cross_modal.py                # 推理、绘图、合并统计
model/BioCross.py                   # BioCross 网络
model/BioCross_base.py              # 基础编码器、解码器
utils/metrics.py                   # 波形误差
utils/phenotype.py                 # 心率和血压
config/TriplemetaVAE3.yaml          # 推理配置
checkpoints/II_Pleth_ABP/model.pth  # 完整模型权重
```

## 服务器运行

建议 Python 3.10+，在仓库根目录执行。GPU 环境应先安装与服务器 CUDA 匹配的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLBACKEND=Agg

python infer_cross_modal.py --task hwd --record 20261116_211644 --target-modality both --age 22 --gender 1
python infer_cross_modal.py --task II_Pleth_ABP --sample-index 0 --target-modality both
python infer_cross_modal.py --help
```

hwd 输入为 `data/hwd/<record>/wave_30hz.csv` 和 `summary.json`；时间从 JSON 读取，年龄和性别由参数提供。II_Pleth_ABP 需 `data/II_Pleth_ABP/Test_data.npy` (N,3,512) 和 `Test_meta.npy` (N,4)。

模型使用 10 秒、512 点窗口，模型端采样率为 51.2 Hz。hwd 的 30 Hz rPPG 会重采样。默认年龄范围 26–91 岁，越界自动截断；男 1、女 0。hwd 没有真实 ECG/ABP 参考，输出指标来自模型生成波形。

可用 `--checkpoint` 指定权重，`--output-dir` 指定输出目录。默认结果位于 `output/<task>/single_sample/`：重建 `.npy`、带 metadata 和误差标注的 `.png`、合并 HR/BP 的 `*_stats.json`。完整参数说明与示例见脚本顶部。

本分支保留现有权重所需的模型参数结构，不需要训练数据或下游分类权重。

权重下载与放置说明见 [checkpoints/README.md](checkpoints/README.md)。

研究引用：BioCross, Information Fusion 123 (2025), 103302. https://doi.org/10.1016/j.inffus.2025.103302
