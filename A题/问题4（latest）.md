# 问题4：充电策略的经验优选与局部鲁棒优化

> **全局主线：**
>
> \[
> \boxed{
> Q1\text{定义稳定退化与基准EOL}
> \rightarrow
> Q2\text{识别策略参数与退化的关系}
> \rightarrow
> Q3\text{预测已有电池未来50圈}
> \rightarrow
> Q4\text{在时间约束下完成策略优选与局部优化}
> }
> \]

问题4不是重新建立第四套寿命模型，而是把前三问已经定义、验证并冻结的对象用于决策。核心原则是：**先比较真实实验策略，再探索受数据支持的局部新候选；新候选只有通过独立稳定性审计才允许升级为推荐。**

---

# 0. 与前三问统一的术语和口径

## 0.1 `chargetime` 与理论充电时间

令策略参数为

\[
\boldsymbol\theta=(C_1,q,C_2),\qquad q=\frac{Q_1}{100}.
\]

根据C-rate定义，理想恒流条件下从0充至80% SOC的两阶段名义理论时间为

\[
T_{0\rightarrow80}^{\mathrm{theo}}(\boldsymbol\theta)
=60\left(\frac{q}{C_1}+\frac{0.8-q}{C_2}\right)\ \mathrm{min}.
\]

该公式不包括80%～100%的统一1C CC-CV阶段，因此不能称为完整0～100%充电时间。

附件 `cycle_train.csv` 中 `chargetime` 的含义不依靠字段名猜测，而沿用Q1/Q2的全电池、全循环理论—实测一致性审计：

1. 比较逐循环 `chargetime` 与 $T_{0\rightarrow80}^{\mathrm{theo}}$ 的数量级；
2. 检查结论是否在不同策略和重复电池间稳定；
3. 使用80%～100%理想1C补充容量所需的最低时间构造反证下界；
4. 只在整体证据一致时，把 `chargetime` 解释为实际快充至约80% SOC的时间指标。

无论审计结果如何，都不把

\[
chargetime-T_{0\rightarrow80}^{\mathrm{theo}}
\]

解释为80%～100%的真实CC-CV时长。没有逐圈CC时长、CV时长或CV容量时，不构造该阶段的伪代理量。

## 0.2 统一稳定退化建模起点

Q1在预先给定的候选集合中，把 $n_0\in\{11,21,31,41,51\}$ 与线性、单调幂律、中心化单调二次结构一起放入嵌套LOBO选择。v4正式冻结为

\[
\boxed{n_0^*=31,\qquad SOH_i(n)=A_i-B_i(n-31)-C_i(n-31)^2,\quad B_i,C_i\ge0.}
\]

Q4不得根据优化结果重新改变这一EOL结构；Q1基准EOL和Q3部分池化EOL都只作辅助，连续优化主响应仍是直接观测的稳定退化速率。

$n_0^*$ 的含义是“从该位置开始，总体稳健趋势适合由统一下降结构描述”，不是“所有电池从该圈以后逐点严格单调”，也不能被解释成某个电化学过程在该圈绝对结束。

## 0.3 直接观测的稳定退化速率

对电池 $i$，在Q1冻结的稳定区间内定义Theil–Sen斜率

\[
k_i^{\mathrm{stable}}
=\operatorname{median}_{n_0^*\le u<v\le150}
\frac{SOH_{i,v}-SOH_{i,u}}{v-u},
\]

并定义正向退化速率

\[
\boxed{
r_i^{\mathrm{stable}}=-k_i^{\mathrm{stable}}
}
\]

使得

\[
r_i^{\mathrm{stable}}\uparrow\Rightarrow\text{退化更快},
\qquad
r_i^{\mathrm{stable}}\downarrow\Rightarrow\text{退化更慢}.
\]

它来自前150圈直接观测SOH，是Q2和Q4的主要退化响应。

## 0.4 基准EOL只作辅助

Q1/Q3定义的80% EOL为冻结退化结构到 $SOH=0.8$ 的远期交点。由于附件没有真实80%寿命标签，Q4不把EOL点估计作为连续优化主目标，只用于：

- 对已有策略进行同一口径下的辅助描述；
- 比较推荐方案与问题一典型长、短寿命策略；
- 提醒短期预测误差与长期结构不确定性是两类不同风险。

---

# 1. 问题四的任务拆解

问题四分为三层：

1. **现有策略经验优选：** 使用全部真实实验策略的观测充电时间和稳定退化速率构造Pareto前沿；
2. **同类参数位置的局部鲁棒优化：** 仅在Q2主分析的同dataset、同类别参数子集附近搜索新候选；
3. **Q3短期一致性与部署更新：** 用完整电池的OOF预测审计已有策略方向；新策略获得早期循环后再启动个体预测。

输出必须同时回答：

- 已有实验策略中哪些方案在“时间—退化”上非支配；
- 是否存在受数据支持的新参数候选；
- 新候选是否真的优于最佳已有策略；
- 最终推荐是已有策略还是待实验候选；
- 推荐适用于什么电芯、类别、时间预算和参数邻域。

---

# 2. 第一层：九种已有策略的经验Pareto优选

## 2.1 电池级和策略级指标

对电池 $i$，定义前150圈观测充电时间的稳健代表值

\[
T_i^{\mathrm{obs}}
=\operatorname{median}_{1\le n\le150}chargetime_{i,n}.
\]

对策略 $g$，定义

\[
T_g^{\mathrm{obs}}
=\operatorname{median}_{i\in g}T_i^{\mathrm{obs}},
\qquad
R_g
=\operatorname{median}_{i\in g}r_i^{\mathrm{stable}}.
\]

于是每个真实策略对应二维点

\[
\left(T_g^{\mathrm{obs}},R_g\right),
\]

横轴越小表示充电越快，纵轴越小表示稳定退化越慢。

## 2.2 Pareto支配关系

若策略 $g_1$ 满足

\[
T_{g_1}^{\mathrm{obs}}\le T_{g_2}^{\mathrm{obs}},
\qquad
R_{g_1}\le R_{g_2},
\]

且至少一个不等号严格成立，则称 $g_1$ 支配 $g_2$。经验Pareto前沿定义为

\[
\mathcal P_{\mathrm{obs}}
=\left\{g:\nexists h\ne g,\ h\prec g\right\}.
\]

该层使用全部49块电池的前150圈观测，不需要隐藏的151～200圈，也不要求策略参数完整，因此九种策略全部保留。

## 2.3 小样本下的前沿稳定性

同一策略内按电池有放回重采样，重复计算 $T_g^{\mathrm{obs}}$、$R_g$ 和Pareto集合。对每个策略报告进入非支配集合的比例。该比例只表示前沿身份对重复电池波动的稳定性，不解释为显著性检验的 $p$ 值。

## 2.4 第一层的决策作用

经验Pareto是最可靠的推荐来源，因为它完全建立在真实实验策略上。即使后续局部连续优化不可靠，问题四仍能基于该前沿给出已有方案中的条件化推荐：

- 强调更短观测时间时，选择前沿左侧策略；
- 允许更长时间以换取更慢直接观测退化时，选择前沿下侧策略；
- 不使用人为的0.5/0.5加权和掩盖时间与退化的量纲差异。

---

# 3. 第二层：同类策略参数位置附近的局部可行域

## 3.1 为什么不对全部九种策略直接拟合连续响应面

九种策略中可有多种参数完整，但跨dataset或类别的差异会与 $C_1,q,C_2$ 同时变化。连续参数模型的主分析子集应沿用Q2：仅保留同一dataset、同一类别且参数完整的策略位置，以降低批次和类别混杂。

`NEWSTRUCTURE` 只作为数据类别标签，不被外推为所有新结构电池的普遍规律。

## 3.2 继承Q2的重参数化

对0～80% SOC的两阶段倍率函数 $C(s;\boldsymbol\theta)$，定义总体倍率水平

\[
A(\boldsymbol\theta)
=\frac{1}{0.8}\int_0^{0.8}C(s;\boldsymbol\theta)\,\mathrm ds
=\frac{C_1q+C_2(0.8-q)}{0.8}.
\]

以预先给定的SOC分界 $s_0$ 构造低SOC和中高SOC平均倍率暴露

\[
E_L(s_0;\boldsymbol\theta)
=\frac{1}{s_0}\int_0^{s_0}C(s;\boldsymbol\theta)\,\mathrm ds,
\]

\[
E_H(s_0;\boldsymbol\theta)
=\frac{1}{0.8-s_0}\int_{s_0}^{0.8}C(s;\boldsymbol\theta)\,\mathrm ds,
\]

以及倍率分配方向

\[
D(\boldsymbol\theta)=E_H-E_L.
\]

Q4直接继承Q2冻结的策略级模型

\[
r^{\mathrm{stable}}
=\beta_0+\beta_AA^*+\beta_DD^*+\varepsilon,
\]

不重新用EOL拟合一套连续寿命响应面，也不根据Q4结果再次改变 $A,D$ 定义。

## 3.3 双重凸包约束

记同类主分析的真实参数点为

\[
\boldsymbol\theta_j=(C_{1j},q_j,C_{2j}),\qquad j=1,\ldots,G_0.
\]

首先要求候选位于原始参数凸包

\[
\Omega_{\theta}
=\operatorname{conv}\{\boldsymbol\theta_1,\ldots,\boldsymbol\theta_{G_0}\}.
\]

仅满足原始参数凸包仍可能在Q2实际回归使用的 $(A,D)$ 特征空间中形成隐蔽外推，因此再定义映射

\[
\Phi(\boldsymbol\theta)
=\bigl(A(\boldsymbol\theta),D(\boldsymbol\theta)\bigr)
\]

并要求

\[
\Phi(\boldsymbol\theta)
\in
\Omega_{AD}
=\operatorname{conv}\{\Phi(\boldsymbol\theta_1),\ldots,\Phi(\boldsymbol\theta_{G_0})\}.
\]

## 3.4 标准化最近邻距离约束

用真实参数点的均值 $\boldsymbol\mu_\theta$ 和逐维标准差 $\boldsymbol s_\theta$ 标准化

\[
\boldsymbol z(\boldsymbol\theta)
=\operatorname{diag}(\boldsymbol s_\theta)^{-1}
(\boldsymbol\theta-\boldsymbol\mu_\theta).
\]

候选到最近真实策略的距离为

\[
d(\boldsymbol\theta)
=\min_j
\left\|\boldsymbol z(\boldsymbol\theta)-
\boldsymbol z(\boldsymbol\theta_j)\right\|_2.
\]

距离上限不根据优化结果调整，而由真实设计点之间的最大最近邻距离预先确定：

\[
d_{\max}
=\max_j\min_{k\ne j}
\left\|\boldsymbol z(\boldsymbol\theta_j)-
\boldsymbol z(\boldsymbol\theta_k)\right\|_2.
\]

因此局部可行域为

\[
\boxed{
\Omega_{\mathrm{local}}
=\left\{
\boldsymbol\theta\in\Omega_{\theta}:
\Phi(\boldsymbol\theta)\in\Omega_{AD},
d(\boldsymbol\theta)\le d_{\max}
\right\}.
}
\]

这一定义把新策略限制为已有实验邻域中的局部插值，而不是在参数矩形范围内任意拼接三个边界值。

---

# 4. 时间预算与搜索规则

## 4.1 时间预算的条件化定义

先在Q2中审计主分析子集的 $T_{0\rightarrow80}^{\mathrm{theo}}$。若其确实形成近等时设计，则把现有同类策略中最大的名义理论时间定义为预算

\[
T_{\mathrm{budget}}
=\max_jT_{0\rightarrow80}^{\mathrm{theo}}(\boldsymbol\theta_j),
\]

并求解

\[
\min_{\boldsymbol\theta}J_{\mathrm{deg}}(\boldsymbol\theta)
\]

满足

\[
\boldsymbol\theta\in\Omega_{\mathrm{local}},
\qquad
T_{0\rightarrow80}^{\mathrm{theo}}(\boldsymbol\theta)
\le T_{\mathrm{budget}}.
\]

如果近等时结构不成立，则不能强行固定时间预算，应保留

\[
\min_{\boldsymbol\theta}
\left(
T_{0\rightarrow80}^{\mathrm{theo}}(\boldsymbol\theta),
J_{\mathrm{deg}}(\boldsymbol\theta)
\right)
\]

的双目标局部Pareto形式。

## 4.2 透明网格搜索

决策变量只有三维，时间、$A,D$ 与退化预测都可快速计算，因此主方法采用预先固定分辨率的穷举网格：

- $C_1,C_2$ 按实验可执行的倍率步长离散；
- $q$ 按实验可执行的SOC百分比步长离散；
- 依次筛除原始凸包外、$(A,D)$ 凸包外、近邻距离超限和时间超限的候选；
- 对全部剩余候选计算同一鲁棒目标。

不使用NSGA-II等随机遗传算法，以避免在三维小问题中引入不必要的随机性和调参空间。网格分辨率必须在查看最优候选前固定，不能为得到某个结果而事后加密。

---

# 5. Bootstrap鲁棒目标与独立推荐门槛

## 5.1 策略内分层Bootstrap

在每次Bootstrap $b$ 中，对每个策略内部按电池有放回重采样，重新计算策略级稳定退化速率并拟合Q2模型，得到

\[
\widehat r^{(b)}(\boldsymbol\theta)
=\widehat\beta_0^{(b)}
+\widehat\beta_A^{(b)}A^*(\boldsymbol\theta)
+\widehat\beta_D^{(b)}D^*(\boldsymbol\theta).
\]

Bootstrap传播的是同一策略重复电池的随机波动，不能创造新的参数位置，也不能消除主分析策略位置很少这一设计限制。

## 5.2 选择目标

不以点预测均值选取“最漂亮”的候选，而使用较不利分位数

\[
J_{0.90}(\boldsymbol\theta)
=Q_{0.90}
\left(
\widehat r^{(1)}(\boldsymbol\theta),\ldots,
\widehat r^{(B_{\mathrm{sel}})}(\boldsymbol\theta)
\right).
\]

候选为

\[
\boldsymbol\theta^*
=\arg\min_{\boldsymbol\theta\in\mathcal G}
J_{0.90}(\boldsymbol\theta),
\]

其中 $\mathcal G$ 是满足全部局部域和时间约束的可实施网格。

## 5.3 选择与验证分离

用于选择 $\boldsymbol\theta^*$ 的Bootstrap样本不能再作为“候选优于已有策略”的唯一证据。确定候选后，重新生成一批独立Bootstrap作为验证集。

令 $\boldsymbol\theta_E$ 为选择阶段表现最好的已有真实策略，在独立验证样本中定义配对差

\[
\Delta^{(b)}
=\widehat r^{(b)}(\boldsymbol\theta^*)
-\widehat r^{(b)}(\boldsymbol\theta_E).
\]

于是

- $\Delta^{(b)}<0$ 表示候选预测退化更慢；
- $P(\Delta<0)$ 表示候选优于已有策略的Bootstrap概率；
- $Q_{0.90}(\Delta)$ 检查较不利情景下候选是否仍优于已有策略。

## 5.4 预先冻结的新策略推荐规则

新候选只有同时满足下列条件才升级为正式推荐：

1. 独立验证中 $Q_{0.90}(\Delta)<0$；
2. 中位改善
   \[
   \operatorname{median}
   \left[
   \widehat r^{(b)}(\boldsymbol\theta_E)
   -\widehat r^{(b)}(\boldsymbol\theta^*)
   \right]
   \]
   超过Q2策略级LOPO误差分辨率；
3. 候选满足全部双凸包、近邻和时间约束；
4. 理论充电时间风险被明确标注，不能伪装成新策略的实测时间。

若任一条件不满足，则最终优先推荐已有实验策略，并把 $\boldsymbol\theta^*$ 仅称为“待实验验证候选”。规则一旦在运行前确定，不得根据结果是否好看而调整。

---

# 6. Q3在问题四中的正确角色

## 6.1 已有策略的短期未来一致性审计

对拥有真实151～200圈的完整电池，只使用Q3外层LOBO生成的OOF预测，计算真实与预测的未来50圈退化量，例如

\[
r_{i,151:200}^{\mathrm{obs}},
\qquad
r_{i,151:200}^{\mathrm{OOF}}.
\]

按策略汇总后，检查Q2/Q4的 $R_g$ 排序是否与短期未来真实退化及OOF预测方向明显冲突。该分析只作方向一致性审计，不作为真实80% EOL验证。

## 6.2 新策略不能直接输入Q3

Q3是“给定某块电池已有前 $L$ 圈历史，预测其后续SOH”的个体状态模型。一个尚未实验的新 $\boldsymbol\theta$ 没有自身前150圈SOH、IR、温度和充电时间，因此设计阶段不能直接调用Q3输出寿命。

正确部署流程是：

1. Q2/Q4依据策略参数和局部响应面筛选候选；
2. 对候选开展小规模真实循环实验；
3. 获得至少所需早期窗口后，按Q1相同流程清洗SOH；
4. 使用Q3冻结的近期趋势模型更新151～200圈短期预测；
5. 若短期状态与设计阶段预期冲突，停止扩大实验或重新评估策略。

9块测试电池每个policy只有单块电池时，不报告虚构的“测试策略中位数”。策略级Q3审计应使用完整电池的OOF结果。

---

# 7. 推荐输出与题目闭环

## 7.1 已有策略推荐

报告经验Pareto前沿，并按时间偏好给出条件化选择。不能只给一个综合分数而隐藏权衡。

## 7.2 新候选推荐

同时报告：

- $C_1,q,C_2$；
- $T_{0\rightarrow80}^{\mathrm{theo}}$；
- $A,D$；
- 到最近真实策略的标准化距离；
- 退化预测的中位数、90%和95%分位数；
- 与最佳已有策略的配对差分布；
- 是否通过预先冻结的推荐门槛。

## 7.3 与问题一典型长短寿命策略比较

问题一的正式典型长、短寿命策略由冻结的基准EOL口径确定；早期观测指标优良代表只能作为辅助诊断。Q4比较时并列报告观测充电时间、稳定退化速率、基准EOL及其角色，但不把基准EOL解释为真实寿命。

## 7.4 合理性与适用范围

最终必须明确：

- 连续模型只覆盖主分析同类策略参数位置的局部邻域；
- 原始参数凸包和 $(A,D)$ 凸包之外均属于外推；
- 新候选的充电时间只有理论值，必须实验确认；
- 策略内Bootstrap不解决参数位置少和残差自由度低的问题；
- Q3高精度仅针对151～200圈短期SOH，不代表80% EOL准确率；
- 任何新参数组合都必须经过真实电池实验后才能用于工程控制。

---

# 8. 建议图表

## 图Q4-1：九种真实策略的观测时间—稳定退化Pareto图

- 横轴：$T_g^{\mathrm{obs}}$；
- 纵轴：$R_g$；
- 标注Pareto前沿和策略代码；
- 点或误差线展示策略内波动；
- 点大小可表示Bootstrap进入前沿的比例。

## 图Q4-2：局部候选搜索图

- 在 $A-D$ 平面展示真实主分析策略位置；
- 展示同时通过双凸包、近邻距离和时间预算的候选；
- 标注最佳已有策略与待验证候选；
- 不把凸包边界画成全参数空间的适用范围。

## 图Q4-3：候选与最佳已有策略的配对Bootstrap差异

- 横轴使用 $\Delta=\widehat r_{\mathrm{candidate}}-\widehat r_{\mathrm{existing}}$；
- 标出0、$Q_{0.90}(\Delta)$ 和LOPO误差分辨率；
- 直接报告 $P(\Delta<0)$；
- 避免只画两个边际分布而忽略配对关系。

## 表Q4-1：九种已有策略经验比较

至少包含策略、样本量、$T_g^{\mathrm{obs}}$、$R_g$、Pareto身份和Bootstrap前沿频率。

## 表Q4-2：最佳已有策略与局部候选

至少包含参数、理论时间、$A,D$、距离、退化分位数、配对差和推荐状态。

---

# 9. 最终执行流程

1. 读取Q1冻结的清洗后SOH、$n_0^*$、$r_i^{\mathrm{stable}}$ 和基准EOL；
2. 读取Q2冻结的样本口径、$A,D$ 定义、标准化参数、策略级模型和LOPO误差；
3. 仅用前150圈逐循环 `chargetime` 计算电池级与策略级观测时间；
4. 对九种策略构造经验Pareto，并做策略内Bootstrap前沿稳定性审计；
5. 检查主分析同类策略的理论时间是否支持近等时预算；
6. 构造原始参数凸包、$(A,D)$ 特征凸包和标准化最近邻阈值；
7. 按预先固定分辨率生成候选网格，筛除所有越界点；
8. 用第一批分层Bootstrap选择最小 $Q_{0.90}$ 候选；
9. 用独立第二批Bootstrap与最佳已有策略作配对比较；
10. 使用预先冻结的“较不利分位数 + LOPO分辨率”门槛决定是否推荐新候选；
11. 使用完整电池的Q3外层OOF结果进行已有策略短期未来一致性审计；
12. 输出推荐、待验证候选和全部适用边界，结果写入 `results/`，不回填本规范。

---

# 10. 必须避免的错误

1. 不把 `chargetime` 与理论时间的残差解释为80%～100% CC-CV时长；
2. 不把理论0～80%时间写成完整充电时间；
3. 不把Q1/Q3基准EOL当成真实寿命标签或Q4连续优化主目标；
4. 不把九种策略、八种参数完整策略和六个主分析同类参数位置混为一谈；
5. 不把重复电池当成新的策略参数位置；
6. 不在参数长方体内任意组合边界值，必须同时约束原始参数凸包和 $(A,D)$ 凸包；
7. 不仅凭边际 $Q_{0.90}$ 更低就宣称新候选更优，必须做独立配对Bootstrap；
8. 不根据结果反向修改网格分辨率、风险分位数或推荐门槛；
9. 不给没有早期循环历史的新策略直接套用Q3；
10. 不把 `NEWSTRUCTURE` 类别标签解释成已知结构机理或普适新电芯规律。

---

# 11. 一句话方法逻辑

\[
\boxed{
\text{九策略真实Pareto}
\rightarrow
\text{同类六点双凸包局部搜索}
\rightarrow
\text{Bootstrap选择/验证分离}
\rightarrow
\text{LOPO分辨率推荐门槛}
\rightarrow
\text{Q3短期一致性与实验后更新}
}
\]

这套结构保证问题四首先回答“现有方案怎么选”，然后才回答“是否值得尝试新方案”，并把任何新候选严格限制为需要真实实验验证的局部插值假设。
