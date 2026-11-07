# CAMB 的积分和 CUDA / MPS 加速分析

## 范围与本地源码

`../camb-tensor/` 是与当前运行环境对应的 CAMB 2.0.5 源码，提交 `1d1d3d6d09e499291651855d4d8286419030a32b`，并已检出匹配的 `forutils` 子模块。保留该源码作为后续改写工作区；当前尚未修改或用它覆盖已安装的 CAMB。

## 与 gCAMB 的关系

用户指出的 [gCAMB 论文](https://arxiv.org/abs/2509.25110) 是直接加速理论计算的 CAMB 版本，值得优先检查。已将其 `gpuport` 分支检出到 `../gcamb/`（提交 `1080aad444f6f278872d95bbb92d427989af7a27`）。论文及源码表明，这一版本用 OpenACC 加速 `TimeSourcesToCl` 中的 `SourceToTransfers`，即 CMB source 的视线投影；论文明确没有移植 hierarchy 演化。`TransferOut` 中的物质 transfer ODE 仍是 CPU OpenMP。

本项目目前通过 PyCCL 设置 `WantCls=False`、`WantTransfer=True`，因而不进入 gCAMB 的 `TimeSourcesToCl` GPU 分支；`bnt_nautilus.core` 的 `ccl.angular_cl` 使用默认 `l_limber=-1`，即 CCL 的 Limber 投影，不是 gCAMB 的 CMB Bessel 投影。因此不能直接套用论文中的性能数字，也无法把该 GPU 核心直接替换到当前物质功率计算。gCAMB 工作区基于 CAMB 1.5.8，本项目基准为 2.0.5；其 README 的 GPU 编译路径使用 NVIDIA `nvfortran` / OpenACC，当前环境未找到 nvfortran，也未建立 MPS 路径。本次没有构建或实测 gCAMB；这些是论文与源码层面的判断。

如果以后加入 CMB 全谱计算，应重新检查并优先复用 gCAMB；当前 BNT 的 CUDA/MPS 工作继续聚焦其未覆盖的物质 transfer 演化及 PyCCL 投影。

研究对象是 BNT 的 `boltzmann_camb + mead2020_feedback` 调用，以及 `BNT_SIGMA8_NORMALIZATION=camb_matter` 的单红移线性归一化。PyCCL 的 `get_camb_pk_lin` 为最终功率谱产生一组 redshift 和 k 网格；与只计算 z=0 的快速归一化不是同样的工作量。

之前的 cProfile 观测中，一次完整 likelihood 大约 8.3 秒（单次、含 profiler），三个宇宙学对象总共调用 CAMB `calc_transfers` 六次，累计约 19.7 秒。这是 Python 可见的入口时间，**不能证明其中哪一个 Fortran 积分是热点**。当前容器没有 `perf`；本报告源码层级的热点排序是待验证的假设。

## 测到的线性归一化切入点

PyCCL 3.2.1 中 `matter_power_spectrum="camb"` 会让 `compute_linear_power()` 同时调用带 `nonlin=True` 的 CAMB 路径，尽管最初只调用 `sigma8`。可在归一化步骤仅构造线性谱，再由最终宇宙学照常计算 HMcode。

`benchmark_ccl_linear.py` 做了三个预热后的对照，记录于 `ccl_linear_benchmark.json`：参考路径 4.37–4.95 秒，线性路径 3.45–4.47 秒；三个点的最大相对 sigma8 差为 4.68e-9。这只是三个点的单次观测，不能证明所有 prior 点等价。

**兼容性限制**：CCL 3.2.1 的线性 CAMB 路径在 CAMB 2.0.5 下触发 `NonLinear_none` 断言，因为 `set_for_lmax` 启用了非线性模式。该基准仅在脚本作用域内临时设置 `NonLinear_none`，完成后还原，未修改共享库；它不是生产修复。若要采用，需先确定正确的上游兼容性改法，再测试多参数点、MPI 和端到端 likelihood。

先前测过的单红移 matter-only 近似更快，但它改变了采样和 sigma8 积分约定；13 个点的最大 S8 相对误差约 0.00164，并非精确等价。两种方案不可混淆。

## 可以考虑 GPU 化的计算

1. **线性 transfer 的时间演化**：`../camb-tensor/fortran/cmbmain.f90` 的 `TransferOut` 已沿 k 使用 OpenMP 并行，每个 k 进入 `GetTransfer`，再由 `GaugeInterface_EvolveScal` 积分到各红移。`RungeKuttaDP45.f90` 使用自适应 Dormand–Prince；同一轨迹的时间步不能并行，不同 k 和不同 cosmology 可以批处理。GPU 实现需要处理不同 k 的步长、接受/拒绝、状态长度以及重子/光子/中微子不同阶段切换；按 k 或状态分桶、mask 和限制同步频率值得研究。只搬单次 ODE 导数到 GPU 会造成大量小 kernel 和同步，未必有收益。这个方向潜在收益大，但工程量和数值风险也最大。
2. **HMcode 的 sigma(R) 查表**：`../camb-tensor/fortran/halofit.f90` 的 `fill_sigtab` 对 64 个 R 逐个调用 `sigma_integral`；内部 `integrate` 用自适应 Simpson 逐层细分并复用已有积分值。对独立的 `[cosmology, redshift, R, integration-node]` 分批处理，预插值线性谱并在设备上归约是较自然的原型。不同 R 的收敛层数不一样，需要分组或 mask，同时控制端点 t=0/1、插值和外推。不能为了吞吐直接用未经收敛验证的固定节点替代原算法。
3. **HMcode 的 halo 质量积分**：`halofit.f90` 中 `p_1h` 已预先折叠梯形积分权重；对 `[redshift, k, halo_mass]` 批量计算 halo window 再按质量维归约。`HMcode_redshift` 已在 CPU 沿 redshift、k 使用 OpenMP；GPU 原型应测 CPU 单线程及合理多线程，计算加上数据搬运与同步。feedback 的多个 pass、归一化及质量定义必须保持一致。
4. **sigma8 的 k 积分**：`../camb-tensor/fortran/results.f90` 的 `Transfer_Get_sigmas` 是 k 网格上的窗口函数与梯形归约。可把 `k`、redshift 和 cosmology 批量化，使用稳定的小 x 窗口展开。但单点、单红移的数据量很小，而且它发生在耗时的 transfer 演化之后；单独移植很可能得不偿失，适合作为批处理接口与数值验证的起点，而不应声称它是已测瓶颈。
5. **中微子动量积分**：`equations.f90` 的 `Nu_Integrate_L01/L012` 位于演化方程内部，默认优化积分只有三个动量节点。单独调用 GPU 过细；如果移植整套 k 演化，可以把动量维作为融合 kernel 的一部分，同时保持有质量中微子的转换和温度约定。

## 双后端设计和验证

### 首个独立原型（2026-11-05）

`../camb-tensor/tensor_prototypes/sigma8.py` 从 CAMB 已计算的 transfer 网格批量计算 sigma8 和 sigma(R)，支持 PyTorch CPU、CUDA、MPS；没有接入或修改 CAMB/HMcode 本身。多半径例程与 `Transfer_Get_sigmas` 一样使用原始 transfer 网格上的梯形归约，**并不复现 HMcode 的自适应 Simpson、冷物质外推及完整半径范围**。

已在两组 cosmology、三个红移测试 CPU/CUDA float64 和 float32 的 sigma8，对照 CAMB `get_sigma8`；六项测试通过（含批量半径及输入检查），MPS 用例因当前节点不可用而跳过。对 0.5–50 Mpc/h 的半径及 `kmax=50`，CUDA float64 与 CAMB `get_sigmaR` 的最大相对差为 0.001835；后者使用更细的插值积分网格。若误用 `kmax=2`，该半径范围的差异在试验中达到约 12%。HMcode 的表范围为 1e-4–1e3 Mpc/h，因此这一结果不能证明足以替代 HMcode。

`../camb-tensor/tensor_prototypes/hm_sigma.py` 另实现了与 HMcode 相同的变量代换及自适应 Simpson 细分流程，对半径批量计算。使用解析光滑谱、5 个半径时，CPU/CUDA 的 float64 和 float32 均与独立 SciPy 积分在相对误差 5e-4 内一致；MPS 用例跳过。其 `p_lin(k)` 由调用者提供，尚未实现 CAMB 的冷物质谱表、插值及外推，因此也尚未对比真正的 HMcode 输出或接入生产。运行：`pixi run --manifest-path BNT_PRD/pixi.toml python camb-tensor/tensor_prototypes/test_hm_sigma.py -v`。

`sigma_r_tensor_batch_benchmark.json` 记录了一次微基准：单 cosmology 的三个红移复制 32 次，64 个半径、206 个 k 节点、预热后重复计时 30 次；CUDA float64 缓存输入中位约 0.60 ms，包含输入/输出搬运约 0.74 ms，单线程 PyTorch CPU float64 约 10.25 ms。三个红移的 CAMB `get_sigmaR` 为约 0.86 ms，但其积分算法不同，工作量也不是相同的 96 组；不可直接从这些数值推导端到端加速比。三红移的小批次上 CUDA 未比单线程 PyTorch CPU 更快。对应的 CAMB transfer 求解仍约需 4.3 秒；批量复制不是 32 个独立 cosmology 的完整计算，不能证明实际采样吞吐。测试过程中还有其他 CAMB 回归任务，计时只作初探。

当前原型要求整个 batch 共用同一条 `k/h` 网格及 primordial power；它可批处理同一个宇宙学的红移，但不能直接批处理 k 网格不同的多个宇宙学。支持多宇宙学还需要逐样本网格、padding/mask 或受控的重采样，并验证由此产生的误差。

运行原型测试：`pixi run --manifest-path BNT_PRD/pixi.toml python camb-tensor/tensor_prototypes/test_sigma8.py -v`。批量计时：`pixi run --manifest-path BNT_PRD/pixi.toml python camb-tensor/tensor_prototypes/benchmark_sigma8.py --batch-copies 32 --output BNT_PRD/sigma_r_tensor_batch_benchmark.json`。环境安装了 PyTorch；macOS MPS 需要在 Apple Silicon 上另行验证。

CUDA、MPS 共用 PyTorch tensor 实现候选积分，避免只写 CUDA 自定义算子。CUDA 先用 float64 参考；MPS 不支持 float64，必须比较 float32 的振荡积分、BNT 相消与最终 χ²，必要时在 CPU 保留双精度。混合 CPU/GPU 实现要把数据在设备常驻并批处理足够多的 redshift / k / cosmology，否则传输与 kernel 开销可能抵消加速。

建议继续在 `camb-tensor` 内完善 `sigma(R)` 或 `p_1h` 原型，保留原 Fortran；对比原函数输出和边界、极端参数、反馈 pass，然后比较线性/非线性 P(k,z)、sigma8、Cℓ 和最终 log likelihood。误差门槛应分别设定：用户给出的 0.005 S8 目标并不自动保证 P(k) 或似然足够准确。记录后端、设备、数据类型、预热、同步、总耗时及峰值内存。当前尚无完整 HMcode 或 Boltzmann 的 GPU 实现。

## Boltzmann / transfer 演化

这是直接求解 Boltzmann 方程的理论数值计算，不使用 emulator。`benchmark_boltzmann.py` 测量预热后的 CAMB API 分阶段耗时，详见 `boltzmann_benchmark.json`：单红移、kmax=2、191 个 k 的 `get_transfer_functions` 中位数约 0.715 秒；64 红移、kmax=10、198 个 k 的中位数约 1.325 秒。之后的 `calc_power_spectra` 分别约 0.094 和 0.255 毫秒。该阶段计时包含背景与 transfer 初始化，不是 Fortran 内逐函数 profile，也不代表完整的 PyCCL/HMcode likelihood；但它显示应优先调查 transfer 计算，而不是单独移植 sigma8 归约。

`cmbmain.f90:TransferOut` 已在 CPU 按 k 用 OpenMP 并行；每个 k 的 `GetTransfer` 调用 `GaugeInterface_EvolveScal`，其在 tight coupling、中微子状态和 hierarchy 截断等事件之间切换，并使用自适应 Dormand–Prince 4/5 ODE。理论上可将多个 k / cosmology 的状态打包在 GPU：先以 CPU 生成相同背景和初始条件，按 k、有效状态维数及物理阶段分组，在组内张量化 RHS 和各 RK stage，跟踪每个轨迹的独立步长及接受/拒绝；跨事件后重新分组。输出必须对齐原 k 网格和各 transfer redshift。每个时间步仍有顺序依赖，分支发散、频繁同步、动态 hierarchy、背景插值及小 batch 都会抵消性能。

推荐先提取 Fortran RHS 输入输出作为对照，验证逐分量导数及 RK 单步，接着对单 k 全时程比对包括切换的轨迹，再扩展到 k 批量和多个 cosmology；逐层检查 transfer、线性 P(k)、sigma8、最终 P(k,z)/Cℓ 和 likelihood。CUDA float64 优先作为数值参考；MPS float32 需独立判断容差。当前没有完整 Boltzmann GPU 实现或端到端加速结论。

### 单步和真实 RHS 的初步原型

`../camb-tensor/tensor_prototypes/rk45.py` 实现 CAMB Dormand–Prince 的七个 trial stage、默认 mixed-norm 误差与逐样本接受/拒绝判定。测试通过临时 Fortran harness 编译原封不动的 `RungeKuttaDP45.f90`，对不同步长、状态和解析 RHS 逐个比较；CPU/CUDA float64、float32 测试通过。float32 中微小误差估计有相消，测试对误差的相对容差为 20%，这并不证明整个自适应求解足够准确。该原型没有实现步长控制、FSAL、事件或状态切换。

`build_rhs_trace.py` 在 `tensor_prototypes/camb-rhs-trace-*/` 创建源码副本并独立编译，通过环境变量在单线程诊断运行中采集 `derivs` 的状态、导数、约束和阶段标志；不修改 `camb-tensor/fortran/` 或现有安装。`capture_rhs.py` 将跟踪版和安装版的线性谱及 sigma8 对比。当前包含 16 个 cosmology、每个 64 条真实 RHS 记录，其中 12 个采用 prior Latin-hypercube 抽样；这不是 prior 全覆盖。fiducial 样本覆盖 8 个 k、16 种状态维数和 4 组阶段标志。运行前必须设置 `OMP_NUM_THREADS=1`，该跟踪构建不用于生产。

`flat_matter_rhs.py` 进一步在平直 LCDM、光子/无质量中微子 hierarchy 未截断的阶段，计算了背景膨胀率、总密度扰动、总动量、引力约束，以及 eta*k、CDM 密度、重子密度、光子密度和无质量中微子密度的导数；对非紧耦合阶段还实现了重子速度、光子和无质量中微子动量导数。测试覆盖 16 组 cosmology 的真实轨迹，CPU 和全部可见 CUDA 设备的 float64 对照通过。

`massive_neutrino.py` 已移植 CAMB 的单个有质量本征态动量积分，包括显式层级、相对论展开和晚期近似；1024 条轨迹中三者均有覆盖。热中微子 rho/P 使用 CAMB 2.0.5 内置公式的 tensor 实现，并与 Fermi–Dirac 直接积分及 CAMB 记录对比；不是训练出的 emulator。它已与上述约束及非紧耦合动量方程串接进行 CPU/CUDA 双精度验证。

`radiation_hierarchy.py` 实现了平直宇宙、非紧耦合且未移除辐射层级时的光子、偏振及无质量中微子高阶矩导数，包含常规截断闭合及高 kτ 中微子近似；不同样本可以有不同层级长度。16 组 cosmology 中有 119 条符合条件的真实 CAMB 状态，逐分量双精度对照通过。测试使用 trace 提供的 opacity、z 和总动量，尚未证明完整耦合演化。当前进程默认只能枚举 3 个 CUDA 设备，而 `nvidia-smi -L` 显示 3 张 H100 的 7 个 MIG slice；`run_mig_tests.py` 按 MIG UUID 启动 7 个独立进程，并要求每个进程实际看到一个 CUDA 设备，7 个 slice 均通过上述测试。运行：`pixi run --manifest-path BNT_PRD/pixi.toml python camb-tensor/tensor_prototypes/run_mig_tests.py`。

`massive_neutrino_hierarchy.py` 实现了单个有质量本征态的显式动量层级、相对论修正层级和晚期流体近似。1024 条真实轨迹分别覆盖 401 条显式动量演化、652 条相对论修正和 19 条晚期近似；三组统计允许重叠。逐分量对照使用双精度、相对容差 1e-10 与绝对容差 3e-16（少数接近零的导数因相消需要绝对容差），7 个 MIG slice 均通过。lmax=2 的闭合式另有合成输入测试，真实轨迹并未覆盖它。背景、引力约束及晚期近似的 G11/G30 系数仍由 trace 提供，尚未连接完整积分器。

这仍不是闭合的 Boltzmann RHS：当前只覆盖一个有质量本征态，重组背景、opacity、其他背景密度和重子压力由 CAMB 或 trace 提供；非平直、移除辐射层级的分支和完整紧耦合分支还未实现，不能计算完整 transfer 或用于生产。

这些真实轨迹揭示 float32 严格逐分量精度问题：在光子密度导数接近相消时，普通 float32 的最坏相对误差约 1.9%（该导数数值很小），严格容差测试因此标为预期失败，不能推断最终 S8 误差。`compensated.py` 对这一项试验了高低位双 float32 算术：从完整双精度参考输入拆分后，788 条未截断样本的最坏相对误差约 2.5e-10，CPU/CUDA 测试通过。它要求输入的低位从一开始就被保留，且目前只覆盖一个表达式；不能当作完整 float32/MPS 演化已解决。MPS 尚未测试。

`rk45_benchmark.json` 仅使用解析 RHS 的七阶段微基准：每轨迹 100 个状态，192 轨迹下单线程 PyTorch CPU float64 中位约 1.76 ms、CUDA float64 约 0.89 ms；单轨迹 CUDA 反而更慢。它没有真实物理 RHS、不同时间步的分组、设备传输和完整 Boltzmann 流程，不能作为实际加速比。下一步需逐阶段移植产生 z/dgq 的约束、光子/中微子 hierarchy 与 background 插值，随后才有条件把 RK 单步接成真实轨迹并比较完整 transfer。

测试：`pixi run --manifest-path BNT_PRD/pixi.toml python -m unittest discover -s camb-tensor/tensor_prototypes -v`。重建跟踪版后用 `capture_rhs.py --trace-source <构建脚本打印的路径>` 生成数据；此操作使用独立 CAMB 副本，不能把它当作经过生产验证的 CAMB 构建。

修改 CAMB 后，从 `../camb-tensor/` 隔离构建并验证实际导入路径，遵循其 `AGENTS.md` 的构建和测试要求；禁止覆盖共享 CAMB 安装。多机或多 GPU 调度仍可与 MPI 配合，tensor 化不自动实现跨节点通信。
