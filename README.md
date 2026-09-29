# π0.5 History-Conditioned Failure Recovery

这是我基于 [OpenPI](https://github.com/Physical-Intelligence/openpi) 和
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 做的一个机器人学习实验项目。我主要想回答一个具体问题：

> 当物体在执行过程中滑落、当前相机又暂时不可用时，π0.5 能不能根据最近的动作—结果历史改变后续动作，并完成重新抓取？

项目包含历史 token 接口、受控滑落干预、配对闭环评测、恢复轨迹行为蒸馏、8 卡 FSDP 训练和因果审计。当前版本已公开代码、实验协议、开发集与冻结初始状态上的三种子结果。

## 当前状态

- π0.5 历史输入和 history compressor：已完成。
- 无历史、当前帧容量对照、历史模型和 last-valid-frame 对照：已完成。
- 受控滑落＋连续 3 次策略查询双相机遮挡：已完成。
- 恢复轨迹数据、两组学习率筛选、三个训练种子：已完成。
- 冻结最终评测和训练后 probe：已完成并审计。

详细状态见 [`STATUS.md`](STATUS.md)。开发集数字只用于选择学习率；冻结初始状态的多种子结果单独报告。

## 模型输入与输出

每次策略查询使用：

```text
当前双相机图像 + 当前机器人状态 + 任务文本
                    +
最近 8 条动作—结果历史
                    ↓
       History Compressor（16 个 history tokens）
                    ↓
       π0.5 prefix + action expert
                    ↓
            10 × 7 连续动作块
```

一条历史记录包含：

- 两个相机在动作执行前后的 SigLIP 特征；每个相机张量为 `[8, 2, 256, 2048]`；
- 已执行动作、机器人状态和状态变化；
- 记录时间与有效 mask。

History compressor 使用 16 个可学习 query，把最多 8 条历史压缩成 16 个与 π0.5 prefix 同维度的 token。动作 token 可以读取当前图像、任务文本、状态和这些历史 token。

## 受控滑落协议

实验固定 LIBERO-10 task 5，并为每个初始状态构造严格配对的两条轨迹：

```text
相同 reset 和逐步动作前缀
          ↓
检测到物体已被抓起并移动至少 12 mm
          ↓
control：保持物体状态
slip：把物体放回固定 free-joint 位姿并清零速度
          ↓
记录干预后的结果帧
          ↓
连续隐藏双相机 3 次策略查询（15 个控制步）
          ↓
比较继续搬运、返回重抓和最终任务成功率
```

配对审计要求两条轨迹的动作前缀哈希相同、机器人本体状态匹配，并保证 oracle 标签只用于构造干预和离线审计，不进入策略输入。完整合同见
[`docs/controlled_slip_protocol.zh-CN.md`](docs/controlled_slip_protocol.zh-CN.md)。

## 恢复轨迹行为蒸馏

这里的“蒸馏”是轨迹级行为蒸馏，不是 logit 蒸馏：

```text
Teacher B2-LVF：最后一帧有效图像 + 当前状态 → 成功恢复动作块
Student π0.5：全零当前图像 + 不同动作—结果历史 → 预测相同动作块
```

训练样本只保留 control 与 slip 都成功的配对 episode。当前数据包含 26 个成功 pair、52 个物理恢复窗口；每个窗口重复采样 20 次，得到 1,040 个恢复 episode，并与原始 LIBERO 示范混合。最终训练缓存包含 67,293 个训练样本。

## 阶段性结果

### 1. 原始历史模型诊断

在第一轮 10 个初始状态的受控滑落诊断中，R7 历史模型的成功率由 control 的 90% 降到 slip 的 70%。虽然滑落改变了历史输入，但首次盲区动作变化只略高于相同输入重复推理噪声，说明原始策略没有把历史中的滑落结果有效路由到恢复动作。

### 2. 恢复蒸馏配置选择

以下 10 个初始状态仅用于选择学习率：

| 配置 | Control | Slip | 用途 |
|---|---:|---:|---|
| R7-original | 80% | 40% | 未进行恢复训练的历史模型 |
| B2-LVF | 100% | 80% | 解析观测补偿 teacher / 强基线 |
| RD-low-s21 | 90% | 80% | 低学习率恢复蒸馏 |
| RD-high-s21 | 60% | 80% | 高学习率恢复蒸馏 |

低学习率版本在该开发集上把 slip 成功轨迹由 4 条提高到 8 条，同时比高学习率版本少损失正常任务能力，因此被选中进入三种子冻结测试。它目前只追平 B2-LVF，尚未证明学习式历史恢复优于解析补偿。

### 3. 冻结评测与机制审计

RD-low 在三个训练种子上的平均闭环成功率为：

| 配置 | Control | Slip |
|---|---:|---:|
| R7-original（单种子基线） | 80.0% | 50.0% |
| B2-zero（单种子基线） | 100.0% | 80.0% |
| B2-LVF（单种子基线） | 100.0% | 80.0% |
| C1（单种子基线） | 80.0% | 80.0% |
| **RD-low（3 种子）** | **76.7% ± 11.5%** | **83.3% ± 5.8%** |

RD 相对 R7-original 提高了 slip 终局成功率的点估计，但没有明显超过不读历史的 B2-zero 或 B2-LVF。它的首个盲区动作差仍与原始 R7 同量级，15 步内重抓数也没有平均提高。probe 能从历史表示中解码一部分滑落信息，但信息还没有稳定路由到即时恢复动作。

完整表格、逐初始状态成败、配对统计和 probe 见
[`results/recovery_distill_final/结论.md`](results/recovery_distill_final/结论.md)。原始 JSON 位于 [`results/`](results/)。

## 数据划分

| 阶段 | LIBERO 初始状态 | 是否用于选择模型 |
|---|---|---|
| 恢复训练 | 10–39 | 训练数据 |
| 学习率选择 | 40–49 | 是 |
| 冻结最终评测 | 0–9 | 否 |

每个训练配置运行 1,500 步，batch size 为 8，使用 8 张 RTX 5090D 32GB 做 FSDP。π0.5 总参数约 33.59 亿，本轮联合更新 history compressor 和动作专家，共约 4.36 亿可训练参数。

## 项目结构

```text
.
├── openpi_overlays/
│   ├── training/              # 历史接口与训练改动
│   └── evaluation_delta/      # 滑落、遮挡与 LVF 评测改动
├── experiments/               # 数据采集、训练、评测和审计脚本
├── docs/                      # 配对协议与实验设计记录
├── results/                   # 轻量 JSON 和阶段性结论
├── tools/apply_openpi_overlay.py
└── STATUS.md
```

数据集、π0.5 权重、训练 checkpoint、完整视频和逐步轨迹体积较大，不提交到 GitHub。

## 权重发布计划

冻结评测已完成。通过许可证与文件审计后，选定权重将单独发布到 Hugging Face，而不是直接放入 GitHub。预计发布：

- B2（B2-zero/LVF 共用权重）、R7-original 和 RD-low seeds 21–23 的可复现 checkpoint；
- 对应的训练配置、OpenPI 基座 commit、随机种子和评测协议；
- 权重文件 SHA256、Model Card 和原始许可证说明。

不发布中间步 checkpoint 和失败训练产物，避免将未经最终评测的版本标记为正式模型。

## 准备 OpenPI

本项目固定 OpenPI commit：

```text
215abfb217dbac7d5f1273282331b9b1866c0479
```

创建训练和评测两个 checkout：

```bash
git clone https://github.com/Physical-Intelligence/openpi.git openpi-train
git -C openpi-train checkout 215abfb217dbac7d5f1273282331b9b1866c0479
python tools/apply_openpi_overlay.py --openpi openpi-train --mode training

git clone https://github.com/Physical-Intelligence/openpi.git openpi-eval
git -C openpi-eval checkout 215abfb217dbac7d5f1273282331b9b1866c0479
python tools/apply_openpi_overlay.py --openpi openpi-eval --mode evaluation
```

随后按照 OpenPI 官方说明安装环境、下载 π0.5 权重，并准备 LIBERO-10 数据。服务器运行脚本通过以下环境变量读取本地路径：

```bash
export PI05_PYTHON=/absolute/path/to/openpi/.venv/bin/python
export PI05_RUNTIME_ROOT=/absolute/path/to/runtime-root
```

实验 runner 默认要求其工作目录中存在训练/评测 OpenPI checkout、缓存索引和基础 checkpoint。当前公开版保留了实际实验编排与参数，尚未包装成一键下载数据和权重的脚本。

## 主要脚本

| 文件 | 作用 |
|---|---|
| `experiments/controlled_slip_protocol.py` | 配对滑落协议及机器审计 |
| `experiments/run_controlled_slip_matrix.py` | 8 卡运行原始对照矩阵 |
| `experiments/run_recovery_teacher_train_collection.py` | 采集训练集恢复窗口 |
| `experiments/build_recovery_distill_cache.py` | 混合原始示范与恢复数据 |
| `experiments/audit_recovery_distill_cache.py` | 检查同当前输入、不同历史以及标签泄漏 |
| `experiments/run_recovery_distill_campaign.py` | 两组学习率、模型选择与三种子训练 |
| `experiments/run_recovery_distill_evaluation.py` | 8 卡配对闭环评测 |
| `experiments/run_recovery_history_probe.py` | 检查 history 表示是否编码执行结果 |
| `experiments/finalize_recovery_distill_results.py` | 汇总置信区间、配对统计和机制指标 |

## 我学到的点

- 给策略增加历史容量，并不代表策略会主动读取历史中的失败信息。
- “看见物体滑落”和“学会返回重抓”是两个不同问题；恢复动作必须出现在训练分布中。
- 评测历史是否有用，需要让 control/slip 的当前输入保持一致，只改变可以追溯的过去结果。
- 强解析基线很重要。当前学习方法与 B2-zero/B2-LVF 的 slip 结果同量级，还没有证明自己的独特优势。
- 机器人闭环实验需要同时保存逐步轨迹、配对合同、失败日志和结果哈希，单个成功率不足以支持机制结论。

## 当前不足

- 目前只有一个 LIBERO 任务和仿真 teleport slip。
- 最终测试每个种子只有 10 个冻结初始状态，统计区间仍会较宽。
- Teacher 使用 last-valid-frame 补偿，恢复数据覆盖的失败类型有限。
- 尚未在真实机器人、自然滑落、长时间遮挡或跨任务场景中验证。
- 当前结果不支持 SOTA、真实机器人恢复能力或通用长期记忆的表述。

## 后续方向

1. 在 Hugging Face 发布选定 checkpoint、配置、Model Card 和哈希。
2. 扩展到抓空、动作 chunk 丢失、物体碰撞和目标位置变化。
3. 分开表示任务目标、执行进度和环境响应，只更新受到新证据影响的上下文。
4. 与显式 execution-state token、普通恢复数据微调和解析控制器进行同预算比较。

## License

本项目代码采用 MIT License。`openpi_overlays/` 基于 Apache-2.0 的 OpenPI 修改，详见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。模型权重与数据集遵循各自原始许可证。
