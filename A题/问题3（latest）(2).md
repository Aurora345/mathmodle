# 问题3：电池寿命预测
## ——基于Q1清洗/平滑曲线的近期稳健趋势预测与候选增强模型

>
> 问题1负责完成异常处理、SOH重算与基础平滑，并在**SOH-only的15个“稳定起点×低参数结构”候选之间通过嵌套验证选择简单、可解释的基准寿命模型**；v4冻结为 $n_0^*=31$ 的中心化单调二次，不使用IR、温度或充电时间进入正式寿命预测。
>
> 问题3在问题1输出的干净序列与SG平滑SOH曲线上进一步提取**长期趋势、最近30圈趋势、SG一阶/二阶导数、IR动态、温度和充电时间动态以及策略信息**。Trend-PCA-Ridge作为候选增强模型，与简单趋势基线进行留一电池样本外比较；最终模型由验证结果决定。这里用于建模的IR、Tavg、ChargeTime都来自 `cycle_train.csv` 的逐循环值；`battery_summary.csv` 中的 `mean_IR`、`mean_Tavg`、`mean_chargetime` 是完整寿命区间真实平均值，**不进入任何SOH/EOL预测、特征选择或参数调优，只在寿命预测冻结后用于辅助准确性检验**。
>
> 候选建模主线：
>
> $$
> \boxed{
> \text{Q1清洗/平滑曲线}
> \rightarrow
> \text{多尺度趋势+SG导数健康特征}
> \rightarrow
> \text{个体趋势Baseline}
> \rightarrow
> \text{PCA低维残差轨迹}
> \rightarrow
> \text{Ridge个体修正}
> \rightarrow
> SOH_{151:200}
> \rightarrow
> \hat L^{Q3}_{80\%}
> }
> $$

---

# 0. 为什么这样建模

题目给出49块电池，其中40块非测试电池有1～200圈数据，9块测试电池只给到150圈，而且9种策略各有1块测试电池。因此问题3真正可用于监督训练的独立电池只有40块。

当前数据具有三个特点：

1. 独立电池数量少，不能把同一块电池的200个循环当成200个独立样本；
2. 151～200圈SOH是一条50维但高度相关、平滑的时间轨迹；
3. 9种测试策略在训练集中均有同策略非测试电池，但其中4种策略只有2块完整训练电池，因此“同策略平均未来轨迹”可以作为基线，但不宜直接作为主模型强先验。

因此不直接使用LSTM、Transformer或50个独立回归模型，而采用：

$$
\boxed{\text{简单个体趋势基线}+\text{低维残差轨迹修正}}
$$

这样先让电池自身已经表现出的退化趋势承担主要预测，再让多维健康信息学习“简单趋势会错在哪里”。同时，Q1的SG平滑曲线不只用于画图，而在Q3中进一步通过一阶/二阶导数提取局部退化速度与加速度信息。

---

# 2. 第1小问：表征电池健康状态的特征提取

## 2.1 数据预处理与Q1衔接：不重新清洗，但充分利用Q1平滑曲线

问题3直接继承问题1最终处理结果：

- Hampel/MAD识别孤立异常；
- 孤立异常线性插值；
- 根据清洗后Capacity重新计算SOH；
- Q1最终确定的SG平滑SOH曲线；
- 清洗后的IR、Tavg、chargetime序列；
- Q1得到的基准寿命估计和基础退化趋势，作为Q3的Benchmark。

这样Q1、Q3使用同一套数据口径，避免同一块电池在不同问题中出现不同SOH曲线。

特别注意：

> **Q3不从原始异常数据重新拟合一套SOH曲线，而是在Q1已经清洗、重算、平滑后的曲线上继续提取动态特征。**

对于SOH，直接使用Q1最终SG平滑曲线：

$$
\boxed{\widetilde{SOH}_i(n)}
$$

对于IR、Tavg和chargetime，先继承Q1的清洗结果；若需要提取导数，则在Q3中仅使用当前可见的早期区间 $1{:}L$ 做轻量SG平滑。这里始终使用逐循环值，不使用三个完整寿命summary均值：

$$
\widetilde{IR}^{rel}(n),\qquad
\widetilde T(n),\qquad
\widetilde t_{chg}(n)
$$

这里的平滑只用于趋势/导数特征提取，不重新定义原始物理量，也不能使用第 $L+1$ 圈之后的数据。

---

## 2.2 为什么“区间稳健斜率 + SG导数”同时保留

Q1已经使用SG平滑，其优势之一就是能够较好保留局部斜率和曲线形状。因此Q3进一步利用同一平滑曲线的一阶、二阶导数是自然的。

若局部SG窗口内拟合多项式：

$$
p_n(x)=a_0+a_1x+a_2x^2+\cdots+a_px^p
$$

则窗口中心处：

$$
\boxed{p_n'(0)=a_1}
$$

$$
\boxed{p_n''(0)=2a_2}
$$

分别表示局部一阶变化速度和二阶变化速度。[6]

但是：

> **导数更细致，不代表一定比Theil-Sen更稳健。**

SG导数对局部曲率更敏感，Theil-Sen则反映一段区间的稳健平均趋势。因此本文不直接用导数完全替代Theil-Sen，而采用：

$$
\boxed{
\text{长期/近期Theil-Sen趋势}
+
\text{SG局部一阶/二阶导数}
}
$$

再由LOBO验证和消融实验决定哪些特征真正保留。

---

## 2.3 截止第 $L$ 圈的SOH特征

对于早期数据长度：

$$
\boxed{L\in\{50,75,100,125,150\}}
$$

每次只能使用：

$$
1,2,\ldots,L
$$

圈信息。

### （1）当前健康状态

直接从Q1平滑曲线读取：

$$
\boxed{S_L=\widetilde{SOH}(L)}
$$

反映预测起点时的健康水平。

### （2）累计健康水平

$$
\boxed{
AUC_L
=
\frac{1}{L-1}
\int_1^L \widetilde{SOH}(n)\,dn
}
$$

实际计算采用梯形积分。

**【本文构造】**

### （3）全局SOH稳健斜率

在Q1平滑SOH曲线上计算：

$$
\boxed{
k^{global}_{SOH,L}
=
TS\{\widetilde{SOH}(n):1\le n\le L\}
}
$$

Theil-Sen斜率对少量残余波动更稳健。[1]

**【文献方法 [1]】**

### （4）近期SOH稳健斜率

统一取截止点前最近30圈：

$$
W_L=\{\max(1,L-29),\ldots,L\}
$$

定义：

$$
\boxed{
k^{recent}_{SOH,L}
=
TS\{\widetilde{SOH}(n):n\in W_L\}
}
$$

### （5）近期退化变化量

$$
\boxed{
\Delta k_{SOH,L}
=
k^{recent}_{SOH,L}-k^{global}_{SOH,L}
}
$$

若：

$$
\Delta k_{SOH,L}<0
$$

表示近期下降速度比长期平均更负，提示衰减可能正在加快。

### （6）SG局部一阶导数：近期瞬时退化速度

在Q1的SG平滑曲线上计算一阶导数：

$$
d^{(1)}_{SOH}(n)
=
\frac{d\widetilde{SOH}(n)}{dn}
$$

不直接使用边界第 $L$ 圈单点导数，而取截止点前最近20圈的中位数：

$$
V_L=\{\max(1,L-19),\ldots,L\}
$$

$$
\boxed{
v_{SOH,L}
=
\operatorname{median}_{n\in V_L}
d^{(1)}_{SOH}(n)
}
$$

这样既利用SG局部求导能力，又降低边界单点导数的不稳定性。

### （7）SG局部二阶导数：衰减加速度（候选特征）

$$
d^{(2)}_{SOH}(n)
=
\frac{d^2\widetilde{SOH}(n)}{dn^2}
$$

定义：

$$
\boxed{
a_{SOH,L}
=
\operatorname{median}_{n\in V_L}
d^{(2)}_{SOH}(n)
}
$$

若：

$$
a_{SOH,L}<0
$$

说明SOH曲线局部向下弯曲，下降速度可能继续变得更负，即存在“退化加速”的迹象。

由于二阶导数比一阶导数更敏感，$a_{SOH,L}$只作为候选特征，必须通过LOBO消融确认有稳定增益后才能进入最终模型。

---

## 2.4 内阻特征：同样提取“水平—趋势—局部导数”

为减少不同电池初始内阻水平差异，先定义相对内阻：

$$
\boxed{
IR^{rel}(n)
=
\frac{IR(n)}{\operatorname{median}(IR_{1:10})}
}
$$

当前状态使用最近10圈中位数：

$$
\boxed{
IR^{rel}_L
=
\operatorname{median}(IR^{rel}_{L-9:L})
}
$$

再在清洗后的相对内阻序列上提取：

### （1）全局内阻趋势

$$
\boxed{
k^{global}_{IR,L}
=
TS\{IR^{rel}(n):1\le n\le L\}
}
$$

### （2）近期内阻趋势

$$
\boxed{
k^{recent}_{IR,L}
=
TS\{IR^{rel}(n):n\in W_L\}
}
$$

### （3）内阻趋势变化

$$
\boxed{
\Delta k_{IR,L}
=
k^{recent}_{IR,L}-k^{global}_{IR,L}
}
$$

若 $\Delta k_{IR,L}>0$，说明近期内阻增长速度高于长期平均水平。

### （4）SG局部内阻增长速度（候选）

对清洗后的相对内阻做仅限 $1{:}L$ 的SG平滑，得到 $\widetilde{IR}^{rel}(n)$，再定义：

$$
\boxed{
v_{IR,L}
=
\operatorname{median}_{n\in V_L}
\frac{d\widetilde{IR}^{rel}(n)}{dn}
}
$$

IR的二阶导数不作为默认特征；若数据实际非常平滑且验证明显受益，再作为扩展分析。

若个别IR为0等已确认异常点，沿用问题1的修复结果，不在问题3重新判定异常。

---

## 2.5 温度特征：可以求趋势/导数，但比SOH和IR更谨慎

温度同时受到老化状态、充电倍率、策略和工况波动影响，因此不把大量温度导数默认塞入模型。

先保留稳健水平：

$$
\boxed{
T_{med,L}
=
\operatorname{median}(Tavg_{1:L})
}
$$

再提取：

$$
\boxed{k^{global}_{T,L}}
$$

$$
\boxed{k^{recent}_{T,L}}
$$

$$
\boxed{
\Delta k_{T,L}
=
k^{recent}_{T,L}-k^{global}_{T,L}
}
$$

若温度序列经早期区间SG平滑后足够稳定，再加入候选局部导数：

$$
\boxed{
v_{T,L}
=
\operatorname{median}_{n\in V_L}
\frac{d\widetilde T(n)}{dn}
}
$$

最终是否保留 $k_T^{recent},\Delta k_T,v_T$ 由LOBO消融结果决定。

---

## 2.6 充电时间特征

定义稳健水平：

$$
\boxed{
t_{chg,L}
=
\operatorname{median}(chargetime_{1:L})
}
$$

再提取：

$$
\boxed{k^{global}_{chg,L}},\qquad
\boxed{k^{recent}_{chg,L}}
$$

以及：

$$
\boxed{
\Delta k_{chg,L}
=
k^{recent}_{chg,L}-k^{global}_{chg,L}
}
$$

如果清洗后的充电时间序列在SG平滑后较稳定，可加入候选导数：

$$
\boxed{
v_{chg,L}
=
\operatorname{median}_{n\in V_L}
\frac{d\widetilde t_{chg}(n)}{dn}
}
$$

温度和充电时间的导数都不预设一定有用，而由交叉验证决定是否保留。

### 充电时间特征与三阶段协议的关系

赛题完整充电协议包含两段可变快充（$0\sim Q_1$ 的 $C_1$、$Q_1\sim80\%$ 的 $C_2$）以及80%～100%的统一1C CC-CV。前两段名义理论时间为：

$$
\boxed{
T_{0-80,i}^{theo}
=60\left(\frac{q_i}{C_{1i}}+\frac{0.8-q_i}{C_{2i}}\right)
}
$$

`cycle_train.csv` 中 `chargetime` 的实际覆盖范围先继承Q1/Q2的数据内一致性核验口径，而不在Q3里重新猜测。若全电池、全循环核验支持其为实际0～80%快充时间，则Q3把它作为随循环演化的**实际快充时间状态量**；若核验尚未完成，则只称“附件逐循环充电时间指标”，不赋予80%以后阶段的含义。

无论最终字段口径如何，Q3都**不再定义**

$$
chargetime-T_{0-80}^{theo}
$$

为80%～100%的CC-CV代理时长，因为附件没有逐圈给出第三阶段CC时长、CV时长或CV容量。Q3只从原始逐循环 `chargetime` 提取水平、趋势和候选导数；$T_{0-80}^{theo}$ 仅作为名义策略结构特征/解释量。若未来获得循环内电流-电压-时间原始序列，再独立提取真实 $T_{CC}$、$T_{CV}$、$Q_{CV}$ 等第三阶段特征。

---

## 2.7 充电策略信息

### 主策略表示：policy类别

9种测试策略在训练数据中均存在，因此主模型中的策略信息优先采用：

$$
\boxed{Policy\ One\text{-}Hot}
$$

这样不会受到部分电池 $C_1$ 缺失的影响，也不需要人为补值。

### 问题2策略特征：作为扩展验证

对参数完整电池，可以进一步加入问题2得到的：

$$
A,\quad D
$$

其中：

- $A$：0～80% SOC总体倍率水平；
- $D$：中高SOC与低SOC的倍率分配差。

但 $A,D$ 不强制进入最终模型，而由消融实验判断其边际预测价值。

---

## 2.8 候选特征不是全部同时进入最终模型

由于真正独立的训练电池只有40块，不能把所有一阶、二阶导数和多尺度统计量全部塞进模型。

最终遵循：

$$
\boxed{
\text{先定义少量有物理意义的候选特征}
\rightarrow
\text{LOBO/嵌套验证}
\rightarrow
\text{消融筛选}
}
$$

若两个高度相关特征提供的信息近乎重复，且加入后没有稳定降低LOBO误差，则保留解释更直接、计算更稳定的一个。

---

# 3. 第2小问：建立未来SOH衰减预测模型

候选增强模型为：

$$
\boxed{\text{Trend-PCA-Ridge}}
$$

即：

> **个体近期趋势作为基线，PCA压缩未来“预测误差轨迹”，Ridge利用早期健康特征预测个体修正。**

---

# 3.1 第一步：建立个体趋势基线

对于给定早期数据长度 $L$，先使用电池自身近期退化速度做最简单外推。

### Baseline-T：近期Theil-Sen趋势外推（默认稳健基线）

$$
\boxed{
\widehat{SOH}^{(0,T,L)}_i(n)
=
S_{i,L}
+
k^{recent}_{SOH,i,L}(n-L)
}
$$

其中：

$$
n=151,\ldots,200
$$

### Baseline-D：SG导数趋势外推（新增对照）

利用Q1平滑SOH曲线最后20圈的一阶导数中位数：

$$
\boxed{
\widehat{SOH}^{(0,D,L)}_i(n)
=
S_{i,L}
+
v_{SOH,i,L}(n-L)
}
$$

Baseline-D用于直接检验“从Q1的SG拟合曲线求导是否比区间Theil-Sen更适合近期外推”。

不能预先假定导数一定更准确，而应在40块完整电池的样本外验证中比较：

$$
RMSE_{Trend\text{-}TS}
\quad vs\quad
RMSE_{Trend\text{-}Deriv}
$$

若SG导数基线在嵌套LOBO中稳定优于Theil-Sen基线，则主模型的个体趋势项改用 $v_{SOH,L}$；否则继续使用更稳健的 $k^{recent}_{SOH,L}$。

定义最终经验证选中的个体趋势速度为：

$$
\boxed{s_{i,L}^{*}}
$$

于是统一写成：

$$
\boxed{
\widehat{SOH}^{(0,L)}_i(n)
=
S_{i,L}
+
s_{i,L}^{*}(n-L)
}
$$

这样既把Q1的SG导数真正用于Q3，又不会因为导数理论上更“细”就未经验证直接替换稳健斜率。

---

# 3.2 第二步：定义未来残差轨迹

对于40块有真实151～200圈数据的非测试电池，定义：

$$
\boxed{
r_i^{(L)}(n)
=
SOH_i(n)
-
\widehat{SOH}^{(0,L)}_i(n),
\quad n=151,\ldots,200
}
$$

**【本文构造】**

这里机器学习不直接预测完整SOH，而是学习：

> **“按照该电池自身早期趋势直接外推以后，未来还需要怎样修正。”**

这样比从零预测50个SOH点更稳定，也保留了个体趋势的物理解释。

---

# 3.3 第三步：PCA压缩50维残差轨迹

将每块训练电池的残差写成向量：

$$
\mathbf r_i
=
[r_i(151),r_i(152),\ldots,r_i(200)]^T
$$

共有40条50维残差轨迹。

先中心化：

$$
\tilde{\mathbf r}_i
=
\mathbf r_i-\bar{\mathbf r}
$$

协方差矩阵：

$$
\boxed{
\mathbf\Sigma
=
\frac{1}{N-1}
\sum_{i=1}^{N}
\tilde{\mathbf r}_i
\tilde{\mathbf r}_i^T
}
$$

求特征值与特征向量：

$$
\boxed{
\mathbf\Sigma\phi_k
=
\lambda_k\phi_k
}
$$

则：

$$
\boxed{
z_{ik}=\phi_k^T\tilde{\mathbf r}_i}
$$

并有：

$$
\boxed{
\mathbf r_i
\approx
\bar{\mathbf r}
+
\sum_{k=1}^{K}z_{ik}\phi_k
}
$$

PCA思想来自Pearson的经典降维方法。[2]

**【文献方法 [2]】**

---

## 主成分数 \(K\) 如何确定

不提前规定 \(K=2\) 或3。

选择最小的 \(K\)，使累计解释率：

$$
\boxed{
\frac{\sum_{k=1}^{K}\lambda_k}
{\sum_j\lambda_j}
\ge95\%
}
$$

**【本文参数选择规则】**

因为151～200圈全部电池都在相同整数循环网格上，所以优先使用普通PCA，不额外使用FPCA；只有普通PCA无法稳定描述轨迹时才考虑函数型扩展。

---

# 3.4 第四步：Ridge预测PCA得分

组成候选早期健康特征向量。为了清楚区分“基础趋势信息”和“SG导数增强信息”，写成分组形式：

$$
\boxed{
\mathbf x_{i,L}
=
[
\mathbf x^{SOH},
\mathbf x^{IR},
\mathbf x^{T},
\mathbf x^{chg},
\mathbf x^{policy}
]^T
}
$$

其中SOH候选组：

$$
\mathbf x^{SOH}
=
[
S_L,
AUC_L,
k^{global}_{SOH,L},
k^{recent}_{SOH,L},
\Delta k_{SOH,L},
v_{SOH,L},
a_{SOH,L}^{optional}
]
$$

IR候选组：

$$
\mathbf x^{IR}
=
[
IR_L^{rel},
k^{global}_{IR,L},
k^{recent}_{IR,L},
\Delta k_{IR,L},
v_{IR,L}^{optional}
]
$$

温度候选组：

$$
\mathbf x^{T}
=
[
T_{med,L},
k^{global}_{T,L},
k^{recent}_{T,L},
\Delta k_{T,L},
v_{T,L}^{optional}
]
$$

充电时间候选组：

$$
\mathbf x^{chg}
=
[
t_{chg,L},
k^{global}_{chg,L},
k^{recent}_{chg,L},
\Delta k_{chg,L},
v_{chg,L}^{optional}
]
$$

策略组按消融实验决定是否加入Policy One-Hot或问题2得到的 $A,D$。

所有连续特征只使用训练折的均值和标准差进行标准化：

$$
Z_j
=
\frac{x_j-\mu_j^{train}}{\sigma_j^{train}}
$$

对第 $k$ 个PCA得分建立：

$$
\boxed{
z_{ik}
=
\beta_{0k}
+
\mathbf x_i^T\boldsymbol\beta_k
+
\varepsilon_{ik}
}
$$

考虑到独立电池只有40块且SOH、IR、温度、充电时间的水平、斜率与导数之间可能高度相关，采用Ridge：

$$
\boxed{
\hat{\boldsymbol\beta}_k
=
\arg\min_{\boldsymbol\beta}
\left[
\|\mathbf z_k-\mathbf X\boldsymbol\beta\|_2^2
+
\lambda\|\boldsymbol\beta\|_2^2
\right]
}
$$

其闭式解：

$$
\boxed{
\hat{\boldsymbol\beta}_k
=
(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}
\mathbf X^T\mathbf z_k
}
$$

**【文献公式 [3]：Hoerl & Kennard, 1970】**

Ridge主要解决：

- 小样本下系数波动；
- 长期斜率、近期斜率、SG导数之间的相关性；
- SOH、IR、温度、充电时间多维特征之间的共线性。

但Ridge不能替代模型验证，因此最终仍通过嵌套LOBO和消融决定是否保留二阶导数、温度导数、充电时间导数等候选特征。

---

# 3.5 第五步：重构未来SOH曲线

得到：

$$
\hat z_{ik}
$$

以后，预测残差：

$$
\boxed{
\hat{\mathbf r}_i
=
\bar{\mathbf r}
+
\sum_{k=1}^{K}\hat z_{ik}\phi_k
}
$$

最终：

$$
\boxed{
\widehat{SOH}_i(n)
=
S_{i,L}
+
s^{*}_{i,L}(n-L)
+
\hat r_i(n),
\quad n=151,\ldots,200
}
$$

**【本文主模型】**

这条公式由三部分组成：

1. 当前健康水平；
2. 经样本外验证选出的近期退化速度（Theil-Sen或SG导数）；
3. 多维健康状态与动态导数特征学习得到的个体修正。

---

# 3.6 同策略平均未来轨迹只作为Baseline

对于策略 \(p\)，训练电池未来变化为：

$$
\Delta SOH_j(h)
=
SOH_j(150+h)-SOH_j(150)
$$

可以定义：

$$
\boxed{
\mu_p(h)
=
\frac{1}{m_p}
\sum_{j\in p}\Delta SOH_j(h)
}
$$

则策略模板预测：

$$
\boxed{
\widehat{SOH}^{policy}_i(150+h)
=
SOH_i(150)+\mu_p(h)
}
$$

**【本文构造】**

但当前4种策略只有2块完整训练电池，因此在LOBO时某些策略模板只能由1块剩余电池形成，稳定性有限。所以它只作为Baseline，不作为主模型先验。

---

# 4. 第3小问：预测151～200圈SOH，并进一步预测80%寿命

## 4.1 151～200圈预测

用40块非测试电池完成模型选择以后，将最终模型在40块完整训练电池上重新拟合，再对9块 `prediction_test=1` 电池预测：

$$
\boxed{
\widehat{SOH}_{151},\ldots,\widehat{SOH}_{200}
}
$$

每块测试电池输出一条50点预测轨迹。

---


## 4.2 80% EOL沿用Q1的“稳定退化段”结构，不再把初期活化段重新塞回去

赛题定义：

\[
\boxed{SOH=0.8}
\]

为寿命终止条件。

问题1已经把“共同观察窗口”和“稳定退化建模窗口”分开，并在40块完整电池内部通过嵌套验证确定：

\[
\boxed{
n_0^*
}
\]

以及最终EOL结构：

\[
\boxed{
n_0^*=31,\qquad
SOH_i(n)=A_i-B_i(n-31)-C_i(n-31)^2,quad B_i,C_i\ge0.
}
\]

因此Q3在得到151～200圈预测以后，不再使用：

\[
SOH_{1:150}^{obs}
+
\widehat{SOH}_{151:200}^{Q3}
\]

整段重新拟合EOL；而是只使用：

\[
\boxed{
SOH_{n_0^*:150}^{obs}
+
\widehat{SOH}_{151:200}^{Q3}
}
\]

重新估计该测试电池的稳定退化参数。

令 $x_i=\hat L_{80\%,i}^{ind}-31$。由

\[
A_i-B_ix_i-C_ix_i^2=0.8
\]

依次得到

\[
C_ix_i^2+B_ix_i-(A_i-0.8)=0,
\]

故当 $C_i>0$ 时取非负根

\[
\boxed{
\hat L_{80\%,i}^{ind}=31+
\frac{-B_i+\sqrt{B_i^2+4C_i(A_i-0.8)}}{2C_i}
}.
\]

当 $C_i=0,B_i>0$ 时退化为

\[
\hat L_{80\%,i}^{ind}=31+\frac{A_i-0.8}{B_i}.
\]

这样Q1和Q3使用的是同一个稳定退化定义，避免Q1刚排除初期活化，Q3又把它加回长期寿命拟合。

### 4.2.1 仅对九块测试电池进行同策略部分池化

中心化二次的单电池EOL仍可能受局部曲率扰动影响。Q3允许使用40块完整训练电池的151～200圈，因此对测试电池最终EOL在对数尺度上向同策略训练中心收缩：

\[
\boxed{
\log \hat L_i^{final}
=w\log \hat L_i^{ind}
+(1-w)\log \hat L_{p(i)}^{peer}
}
\]

其中 $\hat L_{p(i)}^{peer}$ 只由与电池 $i$ 同策略的训练电池形成，$w$ 必须在外层LOBO的每个训练折内部选择。指数化后：

\[
\boxed{
\hat L_i^{final}
=(\hat L_i^{ind})^w
\cdot(\hat L_{p(i)}^{peer})^{1-w}
}.
\]

v4冻结 $w=0.75$，即几何尺度上的75%个体信息与25%同策略信息。该收缩只用于Q3九块测试电池的最终EOL尾部稳健化，不回填Q1的49块统一基准寿命，也不宣称获得真实EOL监督。

### 4.2.2 未来斜率加速度头只作辅助诊断

记前150圈近期斜率为 $k_i^{past}$、151～200圈真实斜率为 $k_i^{future}$，定义

\[
\Delta k_i=k_i^{future}-k_i^{past}.
\]

候选包括直接延续、全体训练电池均值、同策略均值和Ridge。若全体均值与更复杂候选同时处于one-SE近优集合，按简约原则冻结

\[
\boxed{
\widehat{\Delta k}_i=
\frac{1}{|\mathcal T|}\sum_{j\in\mathcal T}\Delta k_j
}
\]

并有 $\hat k_i^{future}=k_i^{past}+\widehat{\Delta k}_i$。该辅助头仅用于说明近期斜率存在总体加速修正，绝不再作为Power/Quadratic的结构门控器。

---

## 4.3 EOL仍是远距离外推，必须单独报告结构不确定性

当前附件只到150/200圈，没有真实80% EOL，因此：

\[
\hat L_{80\%}^{Q3}
\]

仍是**模型外推寿命**。

对于40块完整电池，可以比较：

- 用真实稳定段 $n_0^*{:}200$ 拟合同一EOL模型得到 $\hat L_i^{200,ref}$；
- 用真实 $n_0^*{:}150$ + Q3预测151～200得到 $\hat L_i^{Q3,pred}$。

定义：

\[
RE_{EOL,i}
=
\frac{|\hat L_i^{Q3,pred}-\hat L_i^{200,ref}|}
{\hat L_i^{200,ref}}.
\]

它只能说明“多看50圈以后同一外推结构有多稳定”，不是对真实EOL的直接验证。

此外必须把两类不确定性分开：

\[
\boxed{
CI_{statistical}
}
\]

表示151～200短期预测残差传播造成的不确定性；

\[
\boxed{
Range_{structural}
}
\]

表示合理稳定起点/线性、幂律和中心化二次结构差异造成的长期模型不确定性。

不能把前者包装成“真实寿命95%置信区间”。

---

## 4.4 Bootstrap只负责统计误差传播

Bootstrap仍可重复抽取训练电池/短期预测残差，得到：

\[
\hat L_i^{(1)},\ldots,\hat L_i^{(B)}
\]

并报告：

\[
Median(\hat L_i),
\qquad
[Q_{0.025},Q_{0.975}].
\]

但这一段区间只反映短期SOH预测噪声传播，不包含“真实后期可能出现knee、长期函数形式可能改变”等结构不确定性。

结构不确定性必须另外以不同合理 $n_0$ 和三类低参数结构的情景范围呈现。

# 5. 第4小问：评价模型精度，并比较不同早期数据长度


# 5.1 主验证：40轮嵌套 Leave-One-Battery-Out

问题3真正需要回答的是：

> **只知道一块新电池前面的SOH历史时，能否预测它151～200圈？**

40块非测试电池都具有真实151～200圈，因此主验证采用40轮外层LOBO：

\[
\boxed{
40\text{轮}\times(39\text{块用于选模}+1\text{块完全留出})
}
\]

不是只做一次“39+1”。

外层第 $i$ 折：

1. 完整留出电池 $i$；
2. 该电池151～200圈在模型选择期间完全不可见；
3. 其余39块完成所有模型结构与超参数选择；
4. 对留出电池只使用允许看到的前 $L$ 圈；
5. 输出 $\widehat{SOH}_{151:200}$；
6. 最后才打开真实 $SOH_{151:200}$ 评分。

这样40块电池每一块都恰好当过一次真正的外层测试电池。虽然最终有：

\[
40\times50=2000
\]

个循环点误差，但统计上仍把：

\[
\boxed{
40\text{个电池级外层测试}
}
\]

作为独立泛化单位，因为同一块电池的50个循环点彼此相关。

### 特别说明：TS_ONLY不是“39块训练几个全局权重”

最终候选中的 `TS_ONLY` 为：

\[
\boxed{
\widehat{SOH}_i(n)
=
S_{i,150}^{rob}
+
k_{i,recent}^{TS}(n-150)
}
\]

其中：

- $S_{i,150}^{rob}$ 来自被预测电池自己141～150圈的SOH中位数；
- $k_{i,recent}^{TS}$ 来自这块电池自己的最近30圈SOH。

因此39块电池的作用是**决定采用TS_ONLY还是M2/M4等增强模型**，并不是训练一套TS全局回归权重。

所以Q3的稳定性重点看：

\[
\boxed{
40\text{个外层折是否稳定选择同一模型结构}
}
\]

而不是比较不存在的“40组TS回归权重”。

对于M2/M4等增强模型，39块训练电池才会参与标准化、PCA、Ridge权重和策略系数等参数估计。

### 防止信息泄漏

每个外层折内部都必须重新完成：

- 可见区间内的SG/导数/斜率/AUC；
- 特征标准化；
- PCA与主成分数；
- Ridge参数；
- 策略特征是否保留；
- 若存在其它候选窗口，也只能在内层选择。

不能先看全体40块的未来真值确定模型，再把同一套LOBO误差称为独立最终精度。

# 5.2 内层模型与超参数选择

在每个外层LOBO训练集内部，不仅选择Ridge参数 \(\lambda\)，还要完成所有会受到验证结果影响的模型选择，包括：

- 近期Theil-Sen趋势还是SG导数趋势；
- 保留哪一组消融特征；
- 是否使用Trend-PCA-Ridge增强；
- PCA主成分数 \(K\)；
- Ridge参数 \(\lambda\)；
- 是否保留二阶导数、温度/充电时间导数以及策略特征。

其中Ridge参数例如通过内层交叉验证选择：

$$
\boxed{
\lambda^*
=
\arg\min_{\lambda}
RMSE_{inner}(\lambda)
}
$$

只有内层完成全部选择以后，外层留出电池才用于一次最终评分。这样形成真正的嵌套验证，避免通过测试电池选择模型结构或超参数。

为避免小样本下因极小误差差异频繁改选复杂模型，内层最终采用一标准误规则。设候选$m$在内层各电池上的均方误差为$e_{im}$，先找平均误差最小的$m_{best}$，再取

$$
\mathcal M_{1SE}=\left\{m:\bar e_m\le \bar e_{m_{best}}+SE(e_{i,m_{best}})\right\},
$$

并在$\mathcal M_{1SE}$中按“纯趋势基线$\rightarrow$较少特征$\rightarrow$较多特征/策略标签”的次序选择最简模型。该规则把可解释性写入预先定义的选型原则，而不是在看到外层结果后人工回退。

---


# 5.3 精度指标：原始SOH误差 + 百分比/百分点同时报告

主指标仍为：

\[
MAE
=
\frac1M\sum_{m=1}^{M}|SOH_m-\widehat{SOH}_m|
\]

\[
RMSE
=
\sqrt{
\frac1M\sum_{m=1}^{M}
(SOH_m-\widehat{SOH}_m)^2
}
\]

以及第200圈终点误差：

\[
E_{200,i}
=
|SOH_i(200)-\widehat{SOH}_i(200)|.
\]

为了直观表达，把SOH的0～1比例量同时换算成“SOH百分点”：

\[
\boxed{
MAE_{pp}=100MAE,
\qquad
RMSE_{pp}=100RMSE
}
\]

正式运行后同时报告原始SOH误差和乘以100后的SOH百分点误差；本规范不预填任何实证数值。

另外可辅助报告标准MAPE：

\[
\boxed{
MAPE_{SOH}
=
\frac1M
\sum_{m=1}^{M}
\left|
\frac{\widehat{SOH}_m-SOH_m}{SOH_m}
\right|
\times100\%
}
\]

并定义仅用于可读展示的：

\[
\boxed{
C_{SOH}=100\%-MAPE_{SOH}
}
\]

称为“SOH相对吻合度”，**不称为回归准确率**。

$R^2$ 仍可报告，但SOH跨电池总体差异会使pooled $R^2$ 看起来非常高，所以论文主结论优先使用MAE、RMSE和电池级误差分布。

### 正式短期结果的记录方式（本MD不预填数值）

未来50圈的最终精度必须来自40轮外层LOBO。正式运行后记录：MAE、RMSE、MAPE、$C_{SOH}=100\%-MAPE$、pooled $R^2$、电池级RMSE中位数/90%分位数/最大值以及第200圈误差。

同时保留以下简单基线作为对照，但**其数值必须重新运行后填写**：

1. Persistence：
   \[
   \widehat{SOH}_{151:200}^{persist}=S_{150}^{rob};
   \]
2. 近期OLS趋势；
3. 近期Theil-Sen趋势；
4. M1/M2/M3/M4等Trend-PCA-Ridge增强候选。

正式结果中可计算相对改进：

\[
\boxed{
Improvement_{RMSE}
=
\frac{RMSE_{baseline}-RMSE_{model}}{RMSE_{baseline}}\times100\%
}
\]

用于说明最终模型相对简单基线是否取得实际增益。模型最终选型仍完全由嵌套LOBO和one-SE规则决定，不在本规范中预设最终一定为TS_ONLY或任何增强模型。需要特别预先规定的是：OLS_ONLY与TS_ONLY具有相同参数复杂度；若二者同时进入one-SE近优集合，则按照本规范中“TS为默认稳健基线”的既定职责优先选择TS_ONLY。该同复杂度裁决规则必须在外层结果揭示前固定，不得依据外层误差反向修改。

# 5.4 不同早期循环长度

统一测试：

$$
\boxed{
L=50,75,100,125,150
}
$$

每次所有特征都只能从1～ \(L\) 圈重新计算，并始终预测同一个目标：

$$
\boxed{151\sim200}
$$

比较：

$$
RMSE_{50},
RMSE_{75},
RMSE_{100},
RMSE_{125},
RMSE_{150}
$$

以及对应MAE和 \(E_{200}\)。

这样可以回答：

> 增加多少早期循环数据后，预测收益开始变小？

若这里只是分别报告不同 \(L\) 下的样本外误差，可以并列比较；如果还要进一步从这些 \(L\) 中选择一个“最终最佳早期长度”，则该选择也应放在外层LOBO的内层训练数据中完成，不能看完外层测试误差以后再挑最优 \(L\)。

---

# 5.5 扩展稳健性：Leave-One-Policy-Out

LOPO不作为主评价，而作为更严格压力测试：

$$
\boxed{\text{LOPO}}
$$

每次完整留下一个policy，看模型能否预测从未见过的策略。

如果使用policy one-hot，在LOPO中该策略属于未见类别，因此LOPO更适合检验不依赖策略标签的健康特征模型，或参数完整子集上的 \(A,D\) 模型。


---


# 5.6 完整寿命状态量的外部一致性验证

这一部分不再写成“用三个均值证明EOL准确率”，而是回答：

> **附件没有真实80% EOL，但给出了完整寿命区间的 `mean_IR`、`mean_Tavg`、`mean_chargetime`。最终预测出来的SOH/EOL轨迹，能否推导出与这些独立长期汇总真值相一致的状态结果？**

严格数据顺序为：

\[
\boxed{
\text{先冻结SOH/EOL}
\rightarrow
\text{逐循环生成IR/Tavg/ChargeTime}
\rightarrow
\text{计算完整寿命均值}
\rightarrow
\text{最后打开summary真值比较}
}
\]

三个summary均值此前不得参与：

- SOH特征构造；
- TS/M2/M4选型；
- $n_0$ 或三类EOL结构选择；
- 超参数调优；
- EOL修正。

否则就从“外部验证”变成“利用完整寿命信息反向校准”。

## 5.6.1 逐循环关系：每一个预测SOH都要对应一个预测状态量

对：

\[
X\in\{IR,Tavg,ChargeTime\}
\]

建立：

\[
\boxed{
X_{i,n}
=
f_X(SOH_{i,n},P_i)+\varepsilon_{i,n}
}
\]

其中 $P_i$ 是充电策略。

“一个SOH对应一个状态量”不能理解成所有电池在相同SOH时绝对具有同一个IR/温度/充电时间。更合理的是：

\[
\boxed{
\text{SOH决定相对变化方向，Policy允许改变关系，个体末端观测决定起点}
}
\]

因此保留完整交互关系作为机制检验候选：

\[
X_{i,n}
=
\alpha_{0,X}
+\alpha_{1,X}SOH_{i,n}
+\gamma_{X,P_i}
+\delta_{X,P_i}SOH_{i,n}
+\varepsilon_{i,n}.
\]

但实际轨迹生成优先使用150圈附近的个体锚定形式：

\[
X_{i,150}^{rob}
=
\operatorname{median}X_{i,141:150},
\qquad
S_{i,150}^{rob}
=
\operatorname{median}SOH_{i,141:150},
\]

\[
\boxed{
\widehat X_{i,n}
=
X_{i,150}^{rob}
+
\beta_{X,P_i}
\left(
\widehat{SOH}_{i,n}
-
S_{i,150}^{rob}
\right)
}
\]

其中 $\beta_{X,P_i}$ 可以是全局斜率，也可以在样本外验证支持时使用Policy特异斜率。

第三阶段80%～100%的1C CC-CV仍是共同实验协议，不额外编码成新的策略变量；`chargetime` 直接使用附件逐循环实测指标。

---

## 5.6.2 先在40块完整电池上验证状态关系本身

每个状态通道单独做电池级LOBO：

1. 留出第 $i$ 块电池；
2. 用其余39块的前150圈逐循环SOH和状态量拟合关系；
3. 用留出电池141～150圈确定自己的状态锚点；
4. 先输入该电池**真实151～200圈SOH**，预测151～200圈状态量，用来隔离检验 $SOH/Policy\rightarrow X$ 关系；
5. 再输入Q3预测的151～200圈SOH，做端到端检查；
6. 与留出电池真实151～200圈IR/Tavg/ChargeTime比较。

分别报告：

\[
MAE_X,\qquad RMSE_X
\]

以及必要的相对误差百分比。

只有在样本外表现足够稳定的关系，才能对完整寿命状态一致性提供较强支持。若one-SE最终选择简单保持不变基线，则应如实说明：复杂SOH/Policy关系没有获得稳定增量价值，该通道的远期验证证据相应较弱。

---

## 5.6.3 先生成正式SOH轨迹，再逐圈生成三个状态轨迹

对9块测试电池，SOH轨迹必须来自正式模型：

\[
\widehat S_i(n)=
\begin{cases}
SOH_i^{obs}(n), & 1\le n\le150,\\[4pt]
\widehat{SOH}_i^{Q3}(n), & 151\le n\le200,\\[4pt]
\widehat g_i^{EOL}(n), & 201\le n\le \hat L_i.
\end{cases}
\]

这里 $\widehat g_i^{EOL}$ 必须使用Q1最终冻结的**稳定退化结构**，且EOL参数来自：

\[
SOH_{n_0^*:150}^{obs}
+
\widehat{SOH}_{151:200}^{Q3}.
\]

随后对每一圈逐一计算：

\[
\boxed{
\widehat X_i(n)
=
\widehat f_X(\widehat S_i(n),P_i)
}
\]

因此完整寿命均值不是“拿一个寿命数字直接算出来”的，而是：

\[
\boxed{
\hat L
\rightarrow
\widehat{SOH}_{1:\hat L}
\rightarrow
\widehat X_{1:\hat L}
\rightarrow
\widehat{\overline X}^{life}
}
\]

---

## 5.6.4 计算完整寿命区间状态均值

前150圈使用真实状态量，之后使用逐循环预测：

\[
\boxed{
\widehat{\overline X}_i^{life}
=
\frac{
\sum_{n=1}^{150}X_{i,n}^{obs}
+
\sum_{n=151}^{\hat L_i}\widehat X_i(n)
}{
\hat L_i
}
}
\]

分别得到：

\[
\widehat{\overline{IR}}_i^{life},
\qquad
\widehat{\overline{Tavg}}_i^{life},
\qquad
\widehat{\overline{ChargeTime}}_i^{life}.
\]

---

## 5.6.5 最后才与三个真实完整寿命均值比较，并优先用百分比表达

对9块测试电池，最后打开：

\[
\overline{IR}_i^{true},
\qquad
\overline{Tavg}_i^{true},
\qquad
\overline{ChargeTime}_i^{true}.
\]

每个通道都同时报告原始量纲误差：

\[
MAE_X
=
\frac1{9}\sum_i
\left|
\widehat{\overline X}_i^{life}
-
\overline X_i^{true}
\right|
\]

和标准MAPE：

\[
\boxed{
MAPE_X
=
\frac1{9}\sum_{i=1}^{9}
\left|
\frac{
\widehat{\overline X}_i^{life}
-
\overline X_i^{true}
}{
\overline X_i^{true}
}
\right|
\times100\%
}
\]

为了直观展示，可定义：

\[
\boxed{
C_X=100\%-MAPE_X
}
\]

称为：

> **完整寿命状态吻合度**

而不是“EOL准确率”。

分别给出：

\[
C_{IR},\qquad
C_T,\qquad
C_{chg}.
\]

如果需要一个摘要性数字，可以额外写：

\[
C_{overall}
=
\frac{C_{IR}+C_T+C_{chg}}{3},
\]

但必须称为“平均状态吻合度”，不能写成“寿命预测准确率”。

同时建议保留预测均值—真实均值散点图及45°参考线，避免只看一个综合百分数掩盖系统偏差。

---

## 5.6.6 不再把 $0.5\hat L/2\hat L$ 人工伸缩试验作为主要证据

旧代码曾把寿命人为改为：

\[
0.5\hat L,\qquad 2\hat L
\]

并在 $SOH_{200}$ 到0.8之间用均匀插值重新造SOH轨迹。

这个构造会让不同寿命情景经历几乎相同的SOH取值区间；当：

\[
X\approx \alpha+\beta SOH
\]

时，完整寿命平均状态天然会非常接近。因此“三均值对寿命不敏感”有一部分是这个数学构造本身造成的。

新版：

\[
\boxed{
\text{删除/降级这一人工伸缩试验}
}
\]

主验证只使用**正式EOL模型实际产生的SOH轨迹**。

---

## 5.6.7 如何解释三个真实均值

如果：

1. 151～200圈SOH外层LOBO误差很小；
2. 状态映射在40块电池后50圈也有样本外支持；
3. 正式EOL轨迹推导出的三个完整寿命均值与真实summary均值具有较高吻合度；

则可以写：

> **短期SOH预测具有直接样本外精度；长期预测轨迹在独立完整寿命状态汇总量上也具有较好的外部一致性。**

但仍不能写：

> “EOL预测准确率为XX%。”

原因是summary提供的是完整寿命平均状态，不是真实：

\[
L_{80\%}^{true}.
\]

因此三个状态吻合度是**长期外部一致性指标**，可以增强或削弱对EOL合理性的信心，但不能等价成“寿命圈数误差”。

# 6. 第5小问：充电策略信息及早期衰减特征对预测结果的影响


为避免把候选模型名字混淆，三种最常讨论的方案可直接理解为：

| 模型 | 使用的信息 | 作用 |
|---|---|---|
| `TS_ONLY` | 测试电池自己的末端SOH状态 + 最近30圈Theil-Sen斜率 | 直接外推151～200圈，不使用39块训练出的全局权重 |
| `M2` | TS基线 + SOH动态特征 + IR动态特征 | 用39块训练电池学习PCA残差轨迹的Ridge修正 |
| `M4` | M2/M3状态信息 + 温度 + ChargeTime + Policy one-hot | 检验策略标签在健康状态之外是否还有额外预测价值 |

因此“39块训练模型”主要适用于M2/M4等增强模型；对 `TS_ONLY`，39块主要负责在内层验证中决定是否选择它，真正的预测斜率来自留出电池自己已知的近期SOH。

采用**分组消融实验**，保持Trend-PCA-Ridge候选增强框架不变，只逐步改变输入特征。这样可以直接回答“SG导数是否有增益”“IR/温度/充电时间是否有额外信息”“策略是否仍有边际价值”；最终是否采用增强模型仍由其能否超过简单趋势基线决定。用于决定“保留哪一组特征”的消融比较应在外层LOBO的内层训练数据中完成，外层留出电池只用于评价已经选定的完整流程。

## M0：问题1基准模型

$$
\boxed{
M_0
=
\text{Q1简单、可解释的基准寿命/未来退化模型}
}
$$

Q1的M0固定使用**Q1正式验证后选定的SOH-only基准结构**（v4为 $n_0^*=31$ 中心化单调二次），不再保留“SOH+IR+T”的条件分支。

M0用于回答：

> Q3的动态特征与轨迹模型是否真正优于问题1的简单Benchmark？

---

## M1：SOH稳健趋势特征

$$
\boxed{
M_1:
S_L+AUC_L+k_{SOH}^{global}+k_{SOH}^{recent}+\Delta k_{SOH}
}
$$

这是不用SG导数增强的SOH动态基线。

---

## M1D：加入Q1平滑曲线的SG导数

$$
\boxed{
M_{1D}:
M_1+v_{SOH}
}
$$

二阶导数单独再测：

$$
\boxed{
M_{1D2}:
M_{1D}+a_{SOH}
}
$$

如果：

$$
RMSE_{M_{1D}}<RMSE_{M_1}
$$

说明Q1平滑曲线的一阶导数确实捕捉了Theil-Sen区间斜率之外的局部退化信息。

如果加入 $a_{SOH}$ 没有稳定改善，则不保留二阶导数。

---

## M2：加入内阻动态信息

$$
\boxed{
M_2:
M_{1D}
+
IR_L^{rel}
+
k_{IR}^{global}
+
k_{IR}^{recent}
+
\Delta k_{IR}
}
$$

再单独测试：

$$
M_{2D}=M_2+v_{IR}
$$

如果 $v_{IR}$ 没有额外增益，就保留更稳健的内阻水平和趋势特征。

---

## M3：加入温度与充电时间

先加入稳健水平与趋势：

$$
\boxed{
M_3:
M_2
+
T_{med}
+
k_T^{global/recent}
+
t_{chg}
+
k_{chg}^{global/recent}
}
$$

温度与充电时间的局部导数 $v_T,v_{chg}$ 作为候选增量特征单独测试，不默认保留。

若：

$$
RMSE_{M_3}<RMSE_{M_2}
$$

说明温度/充电时间在SOH和IR之外仍提供额外健康信息。

---

## M4：再加入完整策略标签

$$
\boxed{
M_4:
M_3+Policy
}
$$

如果：

$$
RMSE_{M_4}<RMSE_{M_3}
$$

说明即使已经知道早期实际健康状态及其动态变化，policy仍提供额外预测信息。

如果：

$$
RMSE_{M_4}\approx RMSE_{M_3}
$$

则更合理的解释是：

> 充电策略造成的大部分影响已经被前期SOH、IR、温度和充电时间等状态变量吸收，策略标签的边际预测增益有限。

---

## M5：问题2策略物理特征（扩展分析）

只在参数完整电池中比较：

$$
\boxed{
M_5:
M_3+A+D
}
$$

由于M5样本范围与全样本不同，所以不能直接把M5误差与M1～M4全样本误差简单横向比较；必须在同一参数完整子集上重新运行对应模型再比较。

---

## 最终特征保留原则

不以“特征越多越好”为目标，而采用：

$$
\boxed{
\text{预测误差是否稳定下降}
+
\text{不同LOBO折是否稳定}
+
\text{物理解释是否合理}
}
$$

只有同时满足以上条件的导数/趋势特征才进入最终模型。

---

# 7. 为什么不用“策略条件FPCA-Ridge”作为主模型

这个方案有合理之处，但根据当前数据结构不作为主模型：

1. 9种策略的完整训练电池数为2～7块；
2. 其中4种策略只有2块完整训练电池；
3. LOBO留出其中一块后，同策略均值轨迹可能只剩1块电池，方差很大；
4. 151～200全部在统一规则循环网格上，不存在FPCA必须处理的不规则采样问题；
5. 因此普通PCA已经足够完成低维轨迹表示。

所以：

$$
\boxed{
\text{策略均值轨迹}=Baseline,
\qquad
\text{个体趋势+PCA残差修正}=候选增强模型
}
$$

所有复杂方案均以正式样本外验证为准，不预先固化。纯近期Theil-Sen趋势、SG导数趋势以及Trend-PCA-Ridge等候选方案均按第5节所述外层LOBO + 内层模型选择进行比较，最终模型结构由本次正式运行结果决定。

---


# 8. 与问题1、问题2的衔接

## 问题1 → 问题3

问题1提供：

\[
\boxed{
\text{清洗后的逐循环数据}
+
\widetilde{SOH}(n)
+
SOH_{150}
+
r_{SOH}^{stable}
+
AUC
+
n_0^*
+
\hat L^{Q1}_{base}
}
\]

其中：

- $n_0^*$：只用于长期稳定退化/EOL拟合；
- Q3的151～200短期预测仍可使用前150圈全部可见信息，最终模型由嵌套LOBO与one-SE规则在正式重跑时决定；
- $\hat L^{Q1}_{base}$：稳定退化段SOH-only长期基准；
- `mean_IR`、`mean_Tavg`、`mean_chargetime`：完整寿命真值，只在第5.6节最后打开。

因此两问形成：

> **Q1识别稳定退化阶段并给出长期基准结构；Q3验证短期未来SOH并对测试电池更新EOL。**

---

## 问题2 → 问题3

问题2输出：

\[
A,D,E_L,E_H
\]

以及稳定退化速率与策略参数的关系。

Q3把Policy/$A,D$作为候选增量特征，而不是强制加入最终151～200模型。是否保留显式Policy信息，必须由正式嵌套LOBO比较M4与纯趋势/其他增强候选后决定；本规范不预设其增量价值。

在完整寿命状态外部一致性部分，Policy/$A,D$仍可作为 $X=f_X(SOH,\text{策略})$ 的候选策略项，但最终summary真值绝不参与关系选择或EOL校准。

---

## 问题3 → 问题4

Q3无论最终由嵌套LOBO选中哪一个候选模型，本质上解决的都是：

> **“已经观察到某一块具体电池的早期循环历史以后，怎样预测它后续SOH。”**

因此Q3属于**个体状态预测器**，不是仅输入 $(C_1,Q_1,C_2)$ 就能给未实验策略直接输出寿命的纯策略响应面。问题4使用Q3时分两类：

1. **已有实验策略的验证：** 对40块拥有真实151～200圈的完整电池，利用Q3外层LOBO产生的样本外预测，按policy汇总 $\widehat{SOH}_{200}^{OOF}$、未来50圈SOH下降量等，检查Q2/Q4的策略优选方向是否与短期未来SOH一致；
2. **新候选策略的后续更新：** 尚未实验的新 $(C_1,Q_1,C_2)$ 没有该策略电池自身的前150圈历史，因此设计阶段不直接调用Q3给EOL。先由Q2/Q4的局部策略模型筛选；实际获得早期循环数据后，再启动Q3做个体化151～200圈预测与风险更新。

特别注意：9块 `prediction_test=1` 电池每个policy只有1块，因此问题4不写“每个策略的Q3 EOL中位数”。如果需要策略级Q3验证，使用40块完整电池的OOF结果按policy汇总；9块测试电池只作为各策略最终应用示例。

---

# 9. 最终推荐技术路线

1. 继承Q1清洗后的Capacity、SOH、IR、Tavg、chargetime以及SG平滑SOH；
2. 继承Q1在40块完整训练电池上确定的稳定退化起点 $n_0^*$ 与EOL结构；
3. 对Q3短期预测仍取 $L=50,75,100,125,150$，每次只使用当前可见的 $1{:}L$ 圈；
4. 从可见SOH中提取末端状态、全局/近期Theil-Sen斜率、趋势变化量和候选SG导数；
5. 对IR、Tavg、chargetime提取少量有物理意义的动态特征；Policy和Q2的 $A,D$ 只作为候选增量信息；
6. 建立40轮外层LOBO，每次完整留出1块电池；
7. 在其余39块内部完成Baseline、特征组、PCA维数、Ridge参数和是否保留Policy等全部选型；
8. 外层留出电池只用前 $L$ 圈生成151～200圈预测，最后才打开真实151～200评分；
9. 汇总40个电池级外层测试的MAE、RMSE、MAPE、SOH百分点误差、第200圈误差和电池级RMSE分布；
10. 增加Persistence基线与OLS30对照，证明TS增益不是由“SOH本来变化小”造成；
11. 确定最终151～200模型以后，用全部40块训练电池冻结完整流程；
12. 对9块测试电池直接输出151～200圈SOH，不在9块上计算不可得的未来误差；
13. 将 $SOH_{31:150}^{obs}+\widehat{SOH}_{151:200}$ 接入中心化单调二次EOL结构；
14. 得到单电池 $\hat L_i^{ind}$，再用训练折内同策略中心按 $w=0.75$ 完成对数尺度部分池化，形成9块测试电池正式 $\hat L_i^{final}$；
15. 以Global mean $\Delta k$ 作为未来斜率辅助头，不参与EOL结构门控；
16. Bootstrap只报告短期预测误差传播形成的统计区间；
17. 另用合理 $n_0$/三类低参数结构情景报告长期结构不确定性；
18. 在40块完整电池上建立并LOBO验证 $SOH+Policy\rightarrow IR/Tavg/ChargeTime$ 的逐循环关系；
19. 对9块测试电池先由正式EOL模型逐圈生成 $SOH_{201:\hat L}$；
20. 再逐圈生成IR/Tavg/ChargeTime，计算完整寿命区间均值；
21. 最后才打开 `mean_IR`、`mean_Tavg`、`mean_chargetime` 三个真实summary均值；
22. 分别报告各通道MAE、MAPE和完整寿命状态吻合度 $C_X=100\%-MAPE_X$；
23. 如果需要平均百分比，只称“平均状态吻合度”，不称“EOL准确率”；
24. 不再把 $0.5\hat L/2\hat L$ 的均匀SOH插值试验作为主要证据；
25. 明确：151～200圈是**直接样本外预测精度**，EOL只有**结构敏感性 + 长期状态外部一致性**两类间接支持。

---

# 10. 最终需要的图表

建议只保留真正回答题目的图表：

1. **Q1平滑曲线与导数示意图**：典型电池的 $\widetilde{SOH}$、一阶导数 $dSOH/dn$，必要时标出二阶导数对应的加速区间；
2. **趋势Baseline比较图**：近期Theil-Sen趋势 vs SG导数趋势的LOBO RMSE；
3. **特征消融图**：$M_0,M_1,M_{1D},M_2,M_3,M_4$ 的验证RMSE；
4. **PCA累计解释率图**：说明为什么只保留少量主成分；
5. **40块LOBO预测误差分布图**；
6. **9块测试电池151～200圈SOH预测曲线**；
7. **不同早期数据长度 $L$ 与RMSE关系图**；
8. **9块测试电池EOL估计及Bootstrap区间图**；
9. **完整寿命三均值辅助验证图**：分别绘制 $\widehat{\overline{IR}}^{life}$、$\widehat{\overline{Tavg}}^{life}$、$\widehat{\overline{ChargeTime}}^{life}$ 与 `battery_summary` 真实均值的一一对应散点图和45°参考线；
10. 可选：Q1基准 vs Trend-PCA-Ridge 的EOL/第200圈预测差异图。

---


# 11. 正式计算后需要填写的结果与最终模型结构（放在问题3的文件夹里，命名好）

每次正式运行必须保存原始误差与百分比/百分点表示。

### 151～200圈直接精度

至少记录：

- 外层40折模型选择次数；
- MAE与 $100MAE$ 个SOH百分点；
- RMSE与 $100RMSE$ 个SOH百分点；
- 标准MAPE；
- $C_{SOH}=100\%-MAPE$；
- pooled $R^2$；
- 电池级RMSE中位数、90%分位数、最大值；
- 第200圈误差；
- Persistence、OLS30、TS30对照；
- 分policy的电池级误差，检查是否某一策略系统性失效。

### 长期EOL

记录：

- Q1冻结的 $n_0^*=31$ 和中心化单调二次结构；
- 9块测试电池的个体EOL、同策略训练中心和部分池化正式EOL；
- 部分池化权重的40折选择记录与截断稳定性对照；
- Global mean $\Delta k$ 加速度辅助头的LOBO误差及简单基线对照；
- 短期残差Bootstrap统计区间；
- 合理起点/函数形式造成的结构不确定性范围；
- 不把上述区间写成真实EOL已验证置信区间。

### 完整寿命状态外部一致性

对IR/Tavg/ChargeTime分别记录：

- 40块电池状态映射LOBO的MAE/RMSE；
- 9块测试电池完整寿命均值的原始MAE；
- 标准MAPE；
- 状态吻合度
  \[
  C_X=100\%-MAPE_X;
  \]
- Spearman和偏差方向；
- 可选平均状态吻合度 $C_{overall}$，但不命名为EOL准确率。

---

# 12. 参考文献

[1] **Sen P K.** Estimates of the Regression Coefficient Based on Kendall's Tau[J]. *Journal of the American Statistical Association*, 1968, 63(324): 1379-1389. DOI: 10.1080/01621459.1968.10480934.  
**使用位置：** Theil-Sen稳健趋势斜率。

[2] **Pearson K.** On Lines and Planes of Closest Fit to Systems of Points in Space[J]. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 1901, 2(11): 559-572. DOI: 10.1080/14786440109462720.  
**使用位置：** PCA低维轨迹表示的经典方法来源。

[3] **Hoerl A E, Kennard R W.** Ridge Regression: Biased Estimation for Nonorthogonal Problems[J]. *Technometrics*, 1970, 12(1): 55-67. DOI: 10.1080/00401706.1970.10488634.  
**使用位置：** Ridge正则化目标函数与稳定回归估计。

[4] **Efron B.** Bootstrap Methods: Another Look at the Jackknife[J]. *The Annals of Statistics*, 1979, 7(1): 1-26. DOI: 10.1214/aos/1176344552.  
**使用位置：** EOL预测不确定性的Bootstrap评估。

[5] **Severson K A, Attia P M, Jin N, et al.** Data-driven prediction of battery cycle life before capacity degradation[J]. *Nature Energy*, 2019, 4: 383-391. DOI: 10.1038/s41560-019-0356-8.  
**使用位置：** MIT-Stanford数据背景，以及利用早期循环信息预测循环寿命的研究背景。本文不使用公开完整版数据补充赛题未提供标签。

[6] **Savitzky A, Golay M J E.** Smoothing and Differentiation of Data by Simplified Least Squares Procedures[J]. *Analytical Chemistry*, 1964, 36(8): 1627-1639. DOI: 10.1021/ac60214a047.  
**使用位置：** Q1/Q3一致的Savitzky-Golay平滑思想，以及Q3从局部多项式直接提取一阶、二阶导数的依据。

---


# 13. 正式结果记录规范（本MD不预填实证结果）

本文件只定义问题3的预测、验证和外部一致性评价流程。**所有模型选择结论、误差、准确性、状态关系优劣、EOL点估计及三均值吻合度都必须在新版Q1/Q2/Q3管线完整重跑后生成。**

## 13.1 未来151～200圈SOH直接精度

正式结果至少记录：

- 40轮外层LOBO的模型选择次数；
- MAE、RMSE及对应SOH百分点：
  
  \[
  MAE_{pp}=100MAE,\qquad RMSE_{pp}=100RMSE;
  \]
- 标准MAPE与 $C_{SOH}=100\%-MAPE$；
- pooled $R^2$；
- 电池级RMSE中位数、90%分位数和最大值；
- 第200圈误差；
- Persistence、近期OLS、近期Theil-Sen以及M1/M2/M3/M4等候选的对照结果；
- 最终模型相对Persistence等基线的RMSE改善百分比；
- 分Policy误差，检查是否存在特定策略下的系统性失效。

## 13.2 三个实时状态关系

IR、Tavg、ChargeTime分别通过电池级LOBO重新评价，记录：

- oracle-SOH输入下的MAE/RMSE；
- Q3预测SOH输入下的端到端MAE/RMSE；
- one-SE最终保留的关系族；
- 非平凡关系相对Persistence的MSE改善百分比。

不在本规范中保留任何历史运行的状态关系误差。

## 13.3 完整寿命状态外部一致性

对9块测试电池，在SOH/EOL完全冻结后逐圈生成IR、Tavg、ChargeTime，并与三个真实完整寿命均值比较。每个通道记录：

\[
MAE_X,\qquad
MAPE_X=\frac1N\sum_i\left|\frac{\widehat X_i-X_i}{X_i}\right|\times100\%,\qquad
C_X=100\%-MAPE_X.
\]

同时报告Spearman相关和平均偏差方向。若计算三通道平均百分比，只称“平均完整寿命状态吻合度”，不称“EOL预测准确率”。

## 13.4 9块测试电池

9块 `prediction_test=1` 电池没有真实151～200圈SOH，因此只输出正式冻结模型产生的151～200预测和EOL结果，不在这9块上自行计算不可得的未来误差。

> 本规范文档保持无实证结果状态；正式结果统一写入重跑后的结果文件和论文结果章节。
