# Last-valid-frame 对照后置登记

状态：主筛选结果已读取后的探索性对照，不能写成预登记或 held-out 结论。

观测缺失训练筛选的六个 cell 均未达到机制门槛，因此不追加 seed 23/47。为判断历史分支的表现是否可由简单观测保持解释，固定运行 B2 的 last-valid-frame 对照。

- 模型：已经收敛的 B2-drop25 与 B2-drop50 seed11 checkpoint，不重新训练。
- 条件：q1、q3、q6 全部运行，共六个 cell，不按已有结果选择单一 cell。
- 任务与初始状态：task 0、4、5；initial-state 0–9。
- 配对：复用对应 B2 clean rollout 产生的缺失日程。
- 输入：缺失期双相机画面替换为缺失前最后一个有效画面；本体状态保持当前值。
- 比较：相同 B2 checkpoint 下 last-valid-frame 与全零画面的逐 episode 配对差值。
- 标签：`COMPLETED-DEV` 或 `RUNNING`；不升级为正式泛化结果。
