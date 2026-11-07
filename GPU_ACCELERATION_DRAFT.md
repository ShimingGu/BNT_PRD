# BNT pseudo-Cℓ likelihood：CUDA / MPS 加速草稿

## 范围和基准

以本目录的 `bnt_nautilus` 为基准，保持 KiDS 数据、NLA/TATT 模型、scale cut、BNT 变换和 Nautilus likelihood 的统计含义。先比较单次 likelihood 的端到端时间，再比较多参数点的吞吐量；不把局部矩阵乘法的加速比当作总体加速比。CPU 以当前双精度结果为参考。

当前环境使用 Pixi，安装了 PyCCL、CAMB、FAST-PT、Nautilus、mpi4py 和科学计算依赖。`smoke_likelihood.py` 使用 mode `080002`、真实数据及一组固定参数，初始化后计算一个 likelihood，不启动采样。运行方式：

```bash
BNT_DATA_DIR=/arc/home/gsm/KiDS_BNT_Nautilus pixi run python smoke_likelihood.py
```

从本目录运行；换机器时调整数据目录。该脚本输出 likelihood、耗时和 Python cProfile 汇总；cProfile 会增加开销，后续计时需要关闭 profiler 并多次重复。它仅覆盖 NLA、相关 dz 和不裁剪 k 的一个参数点，不代表 TATT 或采样吞吐量。

2026-11-05 在当前 Linux 节点、单线程配置下的首次试跑：Python 3.12.16、PyCCL 3.2.1、CAMB 2.0.5、FAST-PT 4.1.1、Nautilus 1.0.6；MPI 两进程通信通过。真实 KiDS 数据、630 维向量。修改前（迭代归一化）的 BNT 与非 BNT χ² 分别为 630.6756898690002 和 630.6756898690127；固定点的 log likelihood 为 -315.3378449345001。单次 likelihood 在开启 cProfile 时为 10.848 秒。

曾测试的 `n_iter=0` 在相同输入点给出 `S_8_output=0.9324002691`，偏离过大，**现已放弃**。当前采用 `n_iter=1`：根据初始宇宙学的实际 sigma8 校正一次 A_s，并报告最终实际 S_8。该点结果为 `S_8_output=0.7969999999999997`、BNT χ² = 630.6756898690002、log likelihood = -315.3378449345001，与原迭代版本在这个点一致。开启 cProfile 的单次 likelihood 为 8.293 秒；与上面的 10.848 秒均为单次观测，不是稳定基准。应在整个 prior 范围比较输入与输出，而不能由一个点保证其一致。

### 可选的快速线性归一化

`BNT_SIGMA8_NORMALIZATION=camb_matter` 使用 CAMB 的 z=0 线性物质谱估计一次校正所需的 sigma8；最终物理谱仍通过原 PyCCL/CAMB/HMcode 路径计算。默认 `ccl` 路径不变。快速路径当前仅允许平直 LCDM 的默认 `w0=-1, wa=0, m_nu=0.06 eV`，输出文件名前缀为 `fastsigma8_`，避免与默认路径的 checkpoint 混用。

在 12 个 Latin-hypercube prior 点及 fiducial 点上，预测的 S8 最大相对偏差为 0.001640，中位数为 0.001065；快速和参考归一化的中位耗时分别为 0.726 和 4.863 秒，约 6.7 倍。数据和可复现输入在 `sigma8_benchmark.json`，生成脚本为 `benchmark_sigma8.py`。由于 S8 prior 上限为 1，这些点的绝对偏差也小于 0.005；13 点不能保证整个 prior 或后验范围。此比较使用固定 A_s，依据线性振幅缩放估计一次校正后 S8 的偏差。

端到端 fiducial 点试跑得到 `S_8_output=0.7978489638`，与目标 0.797 的绝对偏差为 0.000849；BNT log likelihood 为 -315.3164988191，单次含 profiler 的 likelihood 耗时 5.645 秒。这一试跑与 prior 扫描部分同时进行，计时仅作粗略参考。scale cut、prior 边界、fiducial 模型和后验对这种近似的影响仍需更密集地验证。

CCL 的 Eisenstein–Hu 路径在直接传入 `A_s` 时要求已有 sigma8，不能直接从 A_s 得到本次所需的振幅；这不排除另行实现含振幅的解析近似或 emulator，但本次未验证它们。

整次初始化及一个 likelihood 的 cProfile 为 38.458 秒，其中三个 `bnt_core` 构造累计 32.899 秒；六次 CAMB `calc_transfers` 累计 24.182 秒；三次 `n_0_n_1` 累计 4.823 秒（其中首次建立 Gauss-Legendre 节点 3.384 秒会被缓存）。这些为嵌套累计时间，不能相加，也不能直接视为单次 likelihood 的分项。测试出现指数溢出 warning，但该点最终结果均有限；需要在更广参数范围单独验证。

## 候选改写

1. **先消除重复 CPU 工作**：S8→As 的归一化保留一次校正，快速线性版本需继续验证。`core.py` 的 `n_0_n_1` 对六个 bin 在相同 10,000 个积分点重复计算宇宙学距离。共享积分节点和距离，在所有 bin 上向量化；参数和红移分布变动时只缓存真正不变的部分。`pCl_matrix` 每次读取 21 个 bandpower 文件，可以预加载；对固定的输入 `ell` 网格，可预先把插值和 bandpower 投影合成矩阵。两者都先做 CPU 基线，不假定 GPU 一定更快。
2. **批量角功率谱投影**：`Cl_tomo_mod` 调用 `ccl.angular_cl` 遍历 tomographic bin 对；在数学和精度确认后，考虑把 kernel、P(k,a) 插值以及视线积分以 `[batch, ell, chi, bin]` 的方式分块计算，并利用 bin 对对称性。NLA 与 TATT 的 GG/GI/IG/II 分量分别验证，尤其不能假定 GI 与 IG 可直接交换。PyCCL 的插值、积分、边界条件、非 Limber 行为和偏差约定须逐项对齐。
3. **BNT 和窗口投影**：将 bandpower、重排索引和 BNT 变换常量驻留设备，对多个参数点做 batched matmul。BNT 变换依赖宇宙学和可能变化的 dz，不可把它无条件缓存为全局常量。单个 630 维向量较小，搬运和调度开销可能盖过收益，优先测批量吞吐。
4. **χ²**：对固定协方差预先做 CPU 双精度 Cholesky 分解；批量 likelihood 可以用三角求解和归约。检查矩阵条件数、残差和误差；MPS 精度不足时保留 CPU 双精度实现。先与当前逆矩阵表达式对照。
5. **TATT / FAST-PT**：单独 profile 后评估 FFT、卷积和插值的 tensor 化；区分一次性初始化与每参数点成本。若需要修改第三方库，把相应源码检出在项目目录内，记录来源、版本和补丁；不直接修改共享安装目录。
6. **CAMB / HMcode / PyCCL**：现阶段作为 CPU 参考。要获得端到端大幅加速，还需要其批量 GPU 实现或经科学验证的替代模型；单独把后处理搬到 GPU 不能消除 CPU 求解时间。未经准确性验证不更换物理模型。

## MPI 与 tensor 批处理

MPI 是多进程/多节点调度，tensor 是表达和计算数组的方法。两者可以组合，但不能通过替换 import 直接互换。

- **单 GPU**：若 likelihood 的主要计算链支持批量参数，可用 Nautilus 的 `vectorized` 接口将多个参数点组成 batch，交给一个 GPU worker。先核实参数字典、blobs 的批量形状、输出顺序以及每点异常处理；当前 `loglike_cosmo` 是标量接口。需要端到端确认 Nautilus 的向量化接口与 blobs 行为。
- **混合阶段**：保留 CPU CAMB/PyCCL 的进程池，对生成的中间量批量执行 GPU 投影。测量序列化、主机与设备传输、GPU 等待和内存成本，避免每个 MPI rank 重复初始化同一个 GPU。
- **多 GPU / 多节点**：保留 MPI 或另用明确的分布式调度，将不同 batch 分到设备；规定 rank 到设备的映射。GPU kernel 内部的批计算并不提供跨节点通信。
- 最终是否删除 `mpi4py` 取决于整个调用链和部署需求，而不是只看后处理能否张量化。Nautilus 和 MPI 初始化、副作用需要先解耦，避免 worker 重复初始化整个模块。

## 精度、验证与设备

CUDA 优先使用 float64 与现有 CPU 对照；Apple MPS 不支持 float64，需评估 float32 中 BNT 相消、协方差条件数和 χ² 误差，必要时采用 CPU float64 混合路径。分别对比 Cℓ、pCℓ、BNT 向量、χ² 和 log likelihood，覆盖不同宇宙学、dz、m、IA、scale cut 和谱接近零的情况。用观测误差及参数后验偏移确定科学容差，再决定是否接受 MPS；不能先假定 float32 足够。

针对 CUDA 和 MPS 分别记录设备、精度、batch size、预热、同步后的时间、峰值内存及传输耗时，与相同线程数的 CPU 基线比较。当前节点的 `nvidia-smi` 可看到 H100 MIG 实例；这不保证当前进程有 CUDA 访问权限。本次没有 MPS 设备测试。

## 建议顺序

先保留当前 CPU 版本与固定输入输出；建立逐函数和完整 likelihood 基线。接着做缓存及 CPU 向量化，再做投影的 CUDA float64 原型，验证后扩展批处理与 MPS float32。最后结合 Nautilus 实测决定 MPI 是否保留，并对代表性链进行统计一致性验证。
