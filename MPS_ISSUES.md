# Apple MPS：待本机验证的问题

当前开发节点没有 MPS；本文件记录待验证事项，不声称已在 Apple GPU 上通过。

1. **浮点精度**：PyTorch MPS 不支持 float64。当前 CUDA 双精度是数值参考，不能直接把 tensor 的设备从 CUDA 换成 MPS 并保留该精度。
2. **相消**：Boltzmann 方程中的 `4*z/3 + q_gamma` 可能由两个约 0.3 的量相减得到约 1e-6 的结果。在当前真实 CAMB 状态样本上，普通 float32 的一个很小的导数出现约 1.9% 相对误差。这不是最终 S8 或 likelihood 的误差；完整轨迹尚未验证。
3. **双 float32 实验**：`camb-tensor/tensor_prototypes/compensated.py` 对单个光子表达式保留高低两部分，在 CPU/CUDA 上与记录的 CAMB 状态一致。但只有从状态初始化开始、在每个更新中持续保留低位才可能用于完整演化；把已有 float32 数据拆分无法恢复精度。该方法尚未在 MPS 上实测，也尚未用于完整约束或 ODE。
4. **自适应求解**：RK 误差估计、步长接受/拒绝和阶段切换会放大或累积微小差异。需要比较逐 k 的切换时刻、状态与导数、transfer、P(k,z)、S8、Cℓ 和 log likelihood；只比较单步不够。
5. **算子与性能**：在 Apple Silicon 上确认 `gather`、mask、归约、三角函数以及涉及 shape 变化的阶段处理都实际运行在 MPS，记录显式同步和数据搬运时间。避免为每个 ODE 步单独同步设备；分组与固定形状会影响吞吐。
6. **环境**：当前 `BNT_PRD/pixi.toml` 仅锁定 Linux；Mac 上需单独安装与本机兼容的 PyTorch、CAMB 等依赖，再运行 `camb-tensor/tensor_prototypes` 的 MPS 测试。不要复用 Linux 的二进制。

有质量中微子动量积分和热背景公式已加入测试：热背景公式的普通 float32 在 CPU/CUDA 测试通过，但完整耦合链的严格 float32 检查仍为预期失败。MPS 必须分别实测这两部分和完整链。

当前原型测试入口（在仓库根目录）：

```bash
python -m unittest discover -s camb-tensor/tensor_prototypes -v
```

严格 float32 容差在 CPU/CUDA 上已暴露问题，相关检查标为预期失败；MPS 检查当前并未运行。目标误差需要结合完整宇宙学参数范围和最终观测量决定。
# 新增辐射层级原型

`../camb-tensor/tensor_prototypes/radiation_hierarchy.py` 使用 PyTorch 的 gather、mask 和逐元素计算，尚未在 MPS 上运行。现有验证采用 CUDA/CPU float64；MPS 不支持该精度，后续需以真实轨迹比较 float32 的各层级导数、截断和高 kτ 分支，并进一步验证完整积分和 S8，不能从 CUDA 结果推断 MPS 精度或性能。该模块目前尚未接入完整演化。

`../camb-tensor/tensor_prototypes/massive_neutrino_hierarchy.py` 同样只验证了 CPU/CUDA 双精度；部分接近零的导数存在相消。MPS 上需单独核对显式动量层级、相对论修正和晚期流体近似的 float32 精度与实际设备性能。

