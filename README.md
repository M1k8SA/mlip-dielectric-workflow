# MLIP Ionic Dielectric Workflow

使用已有的 Born 有效电荷和高频电子介电张量，通过 DeepMD 兼容机器学习势计算有限位移力常数，并生成频率相关离子介电张量。

```text
pw.in + BORN + MLIP model
  -> optional MLIP relaxation
  -> phonopy finite-displacement supercells
  -> MLIP forces -> FORCE_SETS
  -> ionic dielectric post-processing
  -> epsil_ion.dat + calc_epsil_parallel.out
```

流程不包含默认模型，也不会自动选择目录中的模型文件。计算时必须显式传入模型路径。

## 功能

- 从 Quantum ESPRESSO `pw.in` 读取周期结构，要求 `ibrav=0`。
- 可选原子/晶胞弛豫，支持 BFGS、LBFGS 和 FIRE。
- 通过 phonopy 生成有限位移超胞并自动组装 `FORCE_SETS`。
- 自动完成 eV/Å 到 phonopy QE 接口所需 Ry/Bohr 的单位转换。
- 默认扣除完美超胞残余力和每个超胞的平均平移漂移力。
- 校验 BORN 原子数、有限力、FORCE_SETS、Γ 点频率和最终数据形状。
- 支持弛豫与力常数阶段使用不同模型。
- 支持弛豫超时、检查点、失败策略和已有弛豫结构复用。
- 提供单任务和 Slurm job array 批量提交脚本。

## 适用范围

当前计算器后端是 `deepmd.calculator.DP`，模型必须能被当前 DeepMD-kit 环境读取。仓库不附带模型权重，也不直接支持 MACE、CHGNet、SevenNet 等其他计算器后端。

MLIP 能量、力和应力计算不调用 Quantum ESPRESSO，因此不需要真实 UPF 文件。`pw.in` 的 `ATOMIC_SPECIES` 仍需包含元素、质量和 UPF 文件名；这些文件名仅作为 ASE 写回 QE 格式时保留的文本标签。

## 获取仓库

```bash
git clone https://github.com/M1k8SA/mlip-dielectric-workflow.git
cd mlip-dielectric-workflow
```

仓库面向 Linux、WSL 和 Linux HPC 集群。不要把一台机器上生成的 `.venv` 直接复制到另一台机器。

## 安装

### 自动建立 CPU 环境

需要系统提供 `bash`、`curl` 和 Python 3。安装脚本会在仓库内创建隔离的 `.venv`，不需要 `sudo`：

```bash
bash setup_environment.sh
```

固定版本包括 DeepMD-kit 3.1.3、PyTorch 2.10.0 CPU、phonopy 4.4.0、ASE 3.29.0、NumPy 2.2.6 和 MPICH 5.0.1.post1。

### 使用已有环境

环境至少需要：

- Python 3；
- DeepMD-kit，且后端与模型格式兼容；
- phonopy；
- ASE；
- NumPy。

直接调用时使用该环境的 Python：

```bash
/path/to/python run_dielectric.py --help
```

## 输入文件

每个计算目录需要：

### `pw.in`

Quantum ESPRESSO 结构输入，必须满足：

- `ibrav=0`；
- 包含 `CELL_PARAMETERS`；
- 包含 `ATOMIC_SPECIES`；
- 包含 `ATOMIC_POSITIONS`。

QE 的电子学、k 点、截断能和赝势路径不会用于 MLIP 力计算。

### `BORN`

phonopy BORN 格式：

1. 第一行是说明文字；
2. 下一行是展平的 3×3 高频电子介电张量；
3. 后续每行是一个原子的 3×3 Born 有效电荷张量，共 9 个数。

Born 电荷的原子顺序必须与 phonopy 原胞一致。默认 `--primitive-axes P` 保留 `pw.in` 原胞和原子顺序。

### MLIP 模型

模型不放入仓库。两个阶段共用模型时使用：

```bash
--model /absolute/path/to/model
```

使用不同模型时分别指定：

```bash
--relax-model /absolute/path/to/relax_model
--force-model /absolute/path/to/force_model
```

## 快速开始

仓库提供一组 BN 示例输入，但不提供模型：

```bash
cp -r examples/bn demo_case
cd demo_case
../run_dielectric.sh \
  --model /absolute/path/to/model \
  --relax-fixed-cell
```

也可以创建自己的计算目录：

```bash
mkdir my_case
cp /path/to/pw.in /path/to/BORN my_case/
cd my_case
../run_dielectric.sh --model /absolute/path/to/model
```

`run_dielectric.sh` 使用仓库中的 `.venv`，但以当前目录作为计算目录，因此同一份仓库可以服务多个独立材料目录。

成功后，计算目录中得到：

- `epsil_ion.dat`：100×19（默认波长网格）数据，第一列为波长，之后为介电张量实部和虚部；
- `calc_epsil_parallel.out`：低频/高频极限张量和后处理日志。

所有中间文件保存在 `dielectric_work/`，包括弛豫检查点、位移超胞、逐超胞力、`phonopy_disp.yaml` 和 `FORCE_SETS`。

## 常用命令

两个阶段使用不同模型：

```bash
../run_dielectric.sh \
  --relax-model /path/to/fast_model \
  --force-model /path/to/accurate_model
```

跳过弛豫：

```bash
../run_dielectric.sh --skip-relax --force-model /path/to/model
```

复用已有 `dielectric_work/relaxation/relaxed_pw.in`：

```bash
../run_dielectric.sh --reuse-relaxed --force-model /path/to/model
```

限制弛豫时间并使用 LBFGS：

```bash
../run_dielectric.sh \
  --model /path/to/model \
  --relax-optimizer lbfgs \
  --relax-max-steps 200 \
  --relax-timeout 1800
```

其他常用参数：

```bash
# 3×3×2 超胞
../run_dielectric.sh --model /path/to/model --dim 3 3 2

# 固定晶胞，只弛豫内部坐标
../run_dielectric.sh --model /path/to/model --relax-fixed-cell

# 修改波长范围、点数和展宽
../run_dielectric.sh --model /path/to/model \
  --wavelength 2,16,100 --relaxation 0.002

# 查看全部参数
../run_dielectric.sh --help
```

## 弛豫失败与检查点

默认 `--on-relax-failure stop`：未收敛或超时后安全停止，不发布新的介电结果，同时保留最后结构、轨迹、日志和失败摘要。

可显式选择：

```bash
--on-relax-failure input  # 使用原始 pw.in 继续
--on-relax-failure last   # 使用最后检查点继续，可能尚未收敛
```

超时在两个优化步之间检查，不会中断正在进行的一次 MLIP 推理。

## Slurm：单个任务

完整工作流使用 [dielectric.slurm](dielectric.slurm)。先根据集群修改文件顶部的 CPU、内存、时限，并按需增加 `--partition`、`--account` 或 `--gres=gpu:1`。

```bash
cd /path/to/case
MODEL=/absolute/path/to/model \
PYTHON=/absolute/path/to/deepmd-env/bin/python \
sbatch /absolute/path/to/mlip-dielectric-workflow/dielectric.slurm \
  --relax-fixed-cell --relax-timeout 72000
```

分别指定模型：

```bash
RELAX_MODEL=/absolute/path/to/relax_model \
FORCE_MODEL=/absolute/path/to/force_model \
PYTHON=/absolute/path/to/deepmd-env/bin/python \
sbatch /absolute/path/to/mlip-dielectric-workflow/dielectric.slurm \
  --relax-fixed-cell
```

让 `--relax-timeout` 小于 Slurm 的 `#SBATCH --time`，为力常数和介电后处理留出时间。GPU 作业还需要与集群 CUDA 驱动匹配的 DeepMD/PyTorch 环境；仅申请 GPU 不会把 CPU 环境转换为 GPU 环境。

## Slurm：批量 job array

建立 `cases.txt`，每个非空、非注释行是一个独立计算目录。相对路径按 `cases.txt` 所在目录解析：

```text
cases/material_001
cases/material_002
cases/material_003
```

提交三个任务，最多同时运行十个 array 元素：

```bash
MODEL=/absolute/path/to/model \
CASE_LIST=/absolute/path/to/cases.txt \
PYTHON=/absolute/path/to/deepmd-env/bin/python \
sbatch --array=0-2%10 \
  /absolute/path/to/mlip-dielectric-workflow/dielectric.slurm \
  --relax-fixed-cell
```

每个任务只写入自己的材料目录，不会共享 `dielectric_work/` 或最终结果。

## 物理一致性注意事项

- BORN 中包含 Born 有效电荷和高频介电张量，它们原则上对应特定结构。
- 如果弛豫明显改变晶胞或内部坐标，应在弛豫后结构上重新计算 BORN；直接沿用旧值是一种近似。
- 无法重新计算 BORN 时，通常优先考虑 `--relax-fixed-cell` 或 `--skip-relax`。
- 用模型 A 弛豫、模型 B 计算力常数，得到的是模型 B 在模型 A 平衡结构处的 Hessian。残余力扣除不能消除两个势能面的全部不一致。
- 显著虚频通常意味着结构、模型适用性、BORN/原子顺序或超胞收敛性需要进一步检查。

## 仓库结构

```text
run_dielectric.py          完整工作流入口
run_dielectric.sh          使用仓库 .venv 的便捷入口
relax_mlip.py              通用 MLIP 弛豫
calc_force.py              单结构 MLIP 力计算
calc_epsil_parallel.py     离子介电后处理
dielectric.slurm           完整流程和 job array
BFGS.slurm                 单独弛豫的 Slurm 模板
setup_environment.sh       CPU 环境安装脚本
examples/bn/               示例 pw.in 与 BORN
```

## 输出格式

默认 `epsil_ion.dat` 每行包含 19 列：

```text
wavelength_um  eps00_real ... eps22_real  eps00_imag ... eps22_imag
```

完整参数说明：

```bash
./run_dielectric.sh --help
```
