options(encoding = "UTF-8")
if (identical(Sys.getlocale("LC_CTYPE"), "C")) {
  suppressWarnings(Sys.setlocale("LC_ALL", "Chinese_China.utf8"))
}
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(ggrepel)
  library(svglite)
  library(ragg)
})

# Figure contract
# Core conclusion: 前150圈可稳定比较早期退化并准确预测151--200圈，但80%寿命仅能远距离外推。
# Archetype: quantitative grid; backend: R only.
# Evidence hierarchy: 预处理 -> 策略差异 -> 参数效应 -> 样本外预测与消融。
# Q4 contract: 真实策略先做Pareto；局部候选必须同时受原始参数域和A-D特征域约束；
# 若配对Bootstrap改进不稳定，则推荐已有实验策略而不是强行推荐新参数。

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[1]) else "scripts/make_figures.R"
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
args <- commandArgs(trailingOnly = TRUE)
res_dir <- if (length(args) >= 1) normalizePath(args[1], winslash = "/", mustWork = TRUE) else file.path(root, "results", "正式重跑_20260816_v4")
q4_dir <- if (length(args) >= 2) normalizePath(args[2], winslash = "/", mustWork = TRUE) else file.path(res_dir, "问题4")
fig_dir <- if (length(args) >= 3) normalizePath(args[3], winslash = "/", mustWork = FALSE) else file.path(root, "论文写在这里，里面有latex模版", "figures_generated_v4")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

font_cn <- "Microsoft YaHei"
palette <- c(
  blue = "#3B6FB6", teal = "#2A9D8F", orange = "#E28E2C",
  red = "#C74B50", purple = "#7B6AA8", grey = "#747474",
  lightblue = "#A9C6E8", lightgrey = "#D8D8D8", dark = "#262626"
)
pc <- function(name) unname(palette[name])

theme_paper <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = font_cn) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35),
      plot.title = element_text(size = base_size + 1.5, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.2, colour = palette["grey"]),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5, colour = palette["dark"]),
      strip.text = element_text(size = base_size - 0.4, face = "bold"),
      strip.background = element_rect(fill = "#F2F4F7", colour = NA),
      legend.title = element_text(size = base_size - 0.3),
      legend.text = element_text(size = base_size - 0.7),
      legend.key.height = grid::unit(3.5, "mm"),
      panel.spacing = grid::unit(5, "mm"),
      plot.margin = margin(6, 8, 6, 6)
    )
}
theme_set(theme_paper())

save_pub <- function(plot, name, width_mm = 170, height_mm = 110, dpi = 400) {
  w <- width_mm / 25.4; h <- height_mm / 25.4
  svglite::svglite(file.path(fig_dir, paste0(name, ".svg")), width = w, height = h)
  print(plot); dev.off()
  grDevices::cairo_pdf(file.path(fig_dir, paste0(name, ".pdf")), width = w, height = h, family = font_cn)
  print(plot); dev.off()
  ragg::agg_png(file.path(fig_dir, paste0(name, ".png")), width = w, height = h, units = "in", res = dpi, background = "white")
  print(plot); dev.off()
}

read_utf8 <- function(...) read.csv(file.path(res_dir, ...), check.names = FALSE, fileEncoding = "UTF-8")
cycle <- read_utf8("问题1", "q1_01_逐循环清洗数据.csv")
battery <- read_utf8("问题1", "q1_03_四十九块电池指标与基准EOL.csv")
policy <- read_utf8("问题1", "q1_08_九种策略分布统计.csv")
validation <- read_utf8("问题1", "q1_04_十五个稳定段结构候选验证汇总.csv")
q1_nested <- read_utf8("问题1", "q1_06_外层40折稳定起点与模型选择.csv")
q1_fits <- read_utf8("问题1", "q1_07_四十九块电池全部候选参数.csv")
q2_soc <- read_utf8("问题2", "q2_09_SOC分界敏感性_双响应.csv")
q3_base <- read_utf8("问题3", "q3_01_四种简单基线比较.csv")
q3_ablation <- read_utf8("问题3", "q3_03_增强模型特征消融.csv")
q3_length <- read_utf8("问题3", "q3_11_简单基线不同早期长度.csv")
q3_pred <- read_utf8("问题3", "q3_14_九块测试电池151_200正式预测.csv")
q3_summary <- read_utf8("问题3", "q3_15_九块测试电池EOL与统计区间.csv")
q3_life_state <- read_utf8("问题3", "q3_21_九块测试电池完整寿命均值对照.csv")
q3_eol_stability <- read_utf8("问题3", "q3_17_四十块电池预测轨迹EOL稳定性.csv")
q3_pool <- read_utf8("问题3", "q3_25_部分池化嵌套LOBO逐电池.csv")
q4_existing <- read.csv(file.path(q4_dir, "q4_01_九策略经验Pareto.csv"), check.names = FALSE, fileEncoding = "UTF-8")
q4_grid <- read.csv(file.path(q4_dir, "q4_03_局部可行网格与选择目标.csv"), check.names = FALSE, fileEncoding = "UTF-8")
q4_compare <- read.csv(file.path(q4_dir, "q4_04_已有最优与局部候选验证比较.csv"), check.names = FALSE, fileEncoding = "UTF-8")
q4_paired <- read.csv(file.path(q4_dir, "q4_08_局部候选配对Bootstrap差异.csv"), check.names = FALSE, fileEncoding = "UTF-8")

short_policy <- function(x) {
  labels <- c(
    "3_6C-80PER_3_6C" = "3.6C-80%-3.6C",
    "80PER_3_6C" = "80%-3.6C",
    "4_8C_80PER_4_8C" = "4.8C-80%-4.8C",
    "5C_67PER_4C_NEWSTRUCTURE" = "5.0C-67%-4.0C\n新结构",
    "5_3C_54PER_4C_NEWSTRUCTURE" = "5.3C-54%-4.0C\n新结构",
    "5_6C_19PER_4_6C_NEWSTRUCTURE" = "5.6C-19%-4.6C\n新结构",
    "3_7C_31PER_5_9C_NEWSTRUCTURE" = "3.7C-31%-5.9C\n新结构",
    "5_6C_36PER_4_3C_NEWSTRUCTURE" = "5.6C-36%-4.3C\n新结构",
    "4_8C_80PER_4_8C_NEWSTRUCTURE" = "4.8C-80%-4.8C\n新结构"
  )
  out <- unname(labels[as.character(x)])
  out[is.na(out)] <- as.character(x)[is.na(out)]
  out
}

# 图1：异常值处理证据。
d1 <- cycle[cycle$battery_id == 1 & cycle$cycle <= 30, ]
d1$capacity_candidate_flag <- tolower(as.character(d1$capacity_candidate)) == "true"
f1 <- ggplot(d1, aes(cycle)) +
  geom_line(aes(y = SOH, colour = "原始SOH"), linewidth = 0.55) +
  geom_point(data = d1[d1$capacity_candidate_flag, ], aes(y = SOH), shape = 21, size = 2.2, stroke = 0.7, fill = "white", colour = palette["red"]) +
  geom_line(aes(y = SOH_smooth, colour = "题目平滑SOH"), linewidth = 0.65, linetype = 3) +
  geom_line(aes(y = SOH_clean, colour = "清洗重算SOH"), linewidth = 0.65) +
  geom_line(aes(y = SOH_sg, colour = "最终SG曲线"), linewidth = 0.85) +
  scale_colour_manual(values = c("原始SOH" = pc("grey"), "题目平滑SOH" = pc("orange"), "清洗重算SOH" = pc("teal"), "最终SG曲线" = pc("blue"))) +
  labs(title = "孤立容量尖峰必须先修正再平滑", subtitle = "1号电池第12圈；空心圆为Hampel/MAD异常候选", x = "循环次数", y = "SOH", colour = NULL) +
  theme(legend.position = "bottom")
save_pub(f1, "fig1_preprocess", 165, 95)

# 图2：九策略中位曲线与IQR。
early <- cycle[cycle$cycle <= 150, ]
curve_rows <- list(); idx <- 1
for (p in unique(early$policy)) {
  ep <- early[early$policy == p, ]
  for (cy in sort(unique(ep$cycle))) {
    v <- ep$SOH_sg[ep$cycle == cy]
    curve_rows[[idx]] <- data.frame(policy = p, cycle = cy, med = median(v), q1 = quantile(v, .25), q3 = quantile(v, .75)); idx <- idx + 1
  }
}
curves <- do.call(rbind, curve_rows)
curves$policy_label <- short_policy(curves$policy)
f2 <- ggplot(curves, aes(cycle, med)) +
  geom_ribbon(aes(ymin = q1, ymax = q3), fill = palette["lightblue"], alpha = .48) +
  geom_line(colour = palette["blue"], linewidth = .65) +
  facet_wrap(~policy_label, ncol = 3, scales = "free_y") +
  labs(title = "九种快充策略的前150圈SOH轨迹", subtitle = "实线为策略中位数，阴影为四分位区间；纵轴按分面局部缩放", x = "循环次数", y = "平滑SOH")
save_pub(f2, "fig2_policy_soh", 180, 150)

# 图3：早期健康指标与线性外推寿命。
battery$policy_label <- short_policy(battery$policy)
ord <- policy$policy[order(policy$life_median)]
battery$policy_label <- factor(battery$policy_label, levels = short_policy(ord))
f3a <- ggplot(battery, aes(policy_label, SOH150)) +
  geom_boxplot(width = .58, outlier.shape = NA, fill = palette["lightblue"], colour = palette["blue"]) +
  geom_point(position = position_jitter(width = .12, seed = 20260814), size = 1.25, alpha = .78, colour = palette["dark"]) + coord_flip() +
  labs(x = NULL, y = expression(SOH[150]), title = "末端健康水平")
f3b <- ggplot(battery, aes(policy_label, stable_rate * 1e4)) +
  geom_boxplot(width = .58, outlier.shape = NA, fill = "#B9DED7", colour = palette["teal"]) +
  geom_point(position = position_jitter(width = .12, seed = 20260814), size = 1.25, alpha = .78, colour = palette["dark"]) + coord_flip() +
  labs(x = NULL, y = expression("稳定退化速率"~(10^{-4}/"圈")), title = "早期SOH衰减速度（越高越快）")
f3 <- f3a | f3b
save_pub(f3, "fig3_early_metrics", 180, 125)

# 图4：稳定起点与结构共同选择；开发集最优与one-SE正式选择分开标注。
validation$model_cn <- factor(
  validation$model,
  levels = c("linear", "power", "centered_quadratic"),
  labels = c("线性", "单调幂律", "中心化单调二次")
)
validation$choice <- "其余候选"
validation$choice[tolower(as.character(validation$best_candidate)) == "true"] <- "开发集最优"
validation$choice[tolower(as.character(validation$selected_one_SE)) == "true"] <- "one-SE正式选择"
f4a <- ggplot(validation, aes(n0, mean_battery_MSE, colour = model_cn, shape = choice)) +
  geom_line(aes(group = model_cn), linewidth = .65, alpha = .8) +
  geom_errorbar(aes(ymin = pmax(mean_battery_MSE - SE_battery_MSE, 1e-9), ymax = mean_battery_MSE + SE_battery_MSE), width = 1.5, linewidth = .35) +
  geom_point(size = 2.8) +
  geom_hline(yintercept = unique(validation$one_SE_threshold), linetype = 3, colour = pc("red"), linewidth = .55) +
  scale_y_log10(labels = label_number(accuracy = 1e-8)) +
  scale_colour_manual(values = c("线性" = pc("grey"), "单调幂律" = pc("blue"), "中心化单调二次" = pc("teal"))) +
  scale_shape_manual(values = c("其余候选" = 16, "开发集最优" = 17, "one-SE正式选择" = 15)) +
  labs(title = "开发阶段候选", x = "稳定段起点 n₀（圈）", y = "电池级平均MSE（对数轴）", colour = NULL, shape = NULL) +
  theme(legend.position = "bottom")
sel_count <- aggregate(held_out_battery ~ selected_n0 + selected_model, q1_nested, length)
names(sel_count)[3] <- "folds"
model_label <- c(linear = "线性", power = "幂律", centered_quadratic = "中心化二次")
# 外层40折均选择中心化二次结构，右图仅保留起点标签，避免长文字跨面板重叠。
sel_count$label <- paste0(sel_count$selected_n0, "圈")
f4b <- ggplot(sel_count, aes(reorder(label, folds), folds, fill = label)) +
  geom_col(width = .58) + geom_text(aes(label = folds), vjust = -0.45, family = font_cn, size = 3) +
  scale_fill_manual(values = c(pc("blue"), pc("lightblue")), guide = "none") +
  scale_y_continuous(breaks = seq(0, 40, 10), limits = c(0, 43), expand = expansion(mult = c(0, .01))) +
  labs(title = "外层结构频次", x = NULL, y = "外层折数") + coord_flip()
f4 <- (f4a | f4b) + plot_layout(widths = c(1.35, .65)) + plot_annotation(
  title = "稳定起点与退化结构的联合选择",
  subtitle = "红虚线：一标准误阈值；方块：正式选择 n₀=31 中心化单调二次。外层38折选31，2折选21"
)
save_pub(f4, "fig4_model_selection", 183, 98)

# 图5：不同策略中心化二次基准估计寿命（对数轴）。
f5 <- ggplot(battery, aes(policy_label, life_150)) +
  geom_boxplot(width = .58, outlier.shape = NA, fill = "#D6E3F4", colour = palette["blue"]) +
  geom_point(position = position_jitter(width = .13, seed = 20260814), size = 1.35, alpha = .82, colour = palette["dark"]) +
  scale_y_log10(labels = label_number(big.mark = ",")) + coord_flip() +
  labs(title = "不同快充策略的基准估计循环寿命", subtitle = "横轴为对数尺度；该结果是n₀=31中心化二次远距离外推，不等同真实EOL", x = NULL, y = "基准估计寿命（圈，对数轴）")
save_pub(f5, "fig5_life_distribution", 175, 120)

# 图6：证据方向一致的典型长、短寿命策略对照。
long_p <- policy$policy[which.max(policy$life_median)]; short_p <- policy$policy[which.min(policy$life_median)]
cmp <- curves[curves$policy %in% c(long_p, short_p), ]
cmp$group <- ifelse(cmp$policy == long_p, "典型长寿命策略", "典型短寿命策略")
cmp$group <- factor(cmp$group, levels = c("典型长寿命策略", "典型短寿命策略"))
f6 <- ggplot(cmp, aes(cycle, med, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = q1, ymax = q3), alpha = .16, colour = NA) + geom_line(linewidth = .9) +
  scale_colour_manual(values = c("典型长寿命策略" = pc("blue"), "典型短寿命策略" = pc("red"))) +
  scale_fill_manual(values = c("典型长寿命策略" = pc("blue"), "典型短寿命策略" = pc("red"))) +
  labs(title = "典型长、短寿命策略的早期SOH轨迹", subtitle = "v4中EOL排序、SOH₁₅₀、稳定退化速率和AUC方向一致", x = "循环次数", y = "策略中位平滑SOH", colour = NULL, fill = NULL) +
  theme(legend.position = "bottom")
save_pub(f6, "fig6_long_short", 165, 95)

# 图7：SOC区间敏感性。
q2_soc_rate <- q2_soc[q2_soc$response == "stable_rate", ]
soc_long <- rbind(
  data.frame(s0 = q2_soc_rate$s0, beta = q2_soc_rate$beta_low, interval = "低SOC区间"),
  data.frame(s0 = q2_soc_rate$s0, beta = q2_soc_rate$beta_high, interval = "中高SOC区间")
)
f7 <- ggplot(soc_long, aes(s0 * 100, beta, colour = interval)) +
  geom_hline(yintercept = 0, linewidth = .35, colour = palette["grey"]) + geom_line(linewidth = .9) + geom_point(size = 2) +
  scale_colour_manual(values = c("低SOC区间" = pc("teal"), "中高SOC区间" = pc("red"))) +
  labs(title = "SOC分界变化不改变稳定退化的关联方向", subtitle = "响应为策略级稳定退化速率；正系数表示更高倍率暴露与更快退化相关", x = "低/中高SOC分界（%）", y = "标准化回归系数", colour = NULL) +
  theme(legend.position = "bottom")
save_pub(f7, "fig7_soc_sensitivity", 160, 95)

# 图8：早期长度与预测误差。
q3_length_ts <- q3_length[q3_length$baseline == "TS", ]
length_long <- rbind(
  data.frame(length = q3_length_ts$length, error = q3_length_ts$MAE, metric = "MAE"),
  data.frame(length = q3_length_ts$length, error = q3_length_ts$RMSE, metric = "RMSE")
)
f8 <- ggplot(length_long, aes(length, error, colour = metric)) +
  geom_line(linewidth = .9) + geom_point(size = 2.2) +
  scale_x_continuous(breaks = q3_length_ts$length) + scale_y_continuous(labels = label_number(accuracy = .0001)) +
  scale_colour_manual(values = c("MAE" = pc("teal"), "RMSE" = pc("blue"))) +
  labs(title = "更多早期循环显著降低预测误差", subtitle = "固定模型：近期Theil--Sen趋势外推；L=150由题设确定", x = "可见早期循环长度（圈）", y = "留一电池误差", colour = NULL) +
  theme(legend.position = "bottom")
save_pub(f8, "fig8_length_accuracy", 160, 95)

# 图9：增强模型消融。
q3_ablation$feature_set <- factor(q3_ablation$feature_set, levels = rev(q3_ablation$feature_set[order(q3_ablation$RMSE)]))
baseline_rmse <- q3_base$RMSE[q3_base$baseline == "TS"]
f9 <- ggplot(q3_ablation, aes(feature_set, RMSE, fill = RMSE < baseline_rmse)) +
  geom_hline(yintercept = baseline_rmse, linetype = 2, colour = palette["red"], linewidth = .6) +
  geom_col(width = .64) + coord_flip() +
  geom_text(aes(label = sprintf("%.6f", RMSE)), hjust = -0.08, family = font_cn, size = 2.8) +
  scale_fill_manual(values = c("TRUE" = pc("teal"), "FALSE" = pc("lightblue")), guide = "none") +
  expand_limits(y = max(q3_ablation$RMSE) * 1.18) +
  labs(title = "PCA--Ridge增强模型未降低主评价RMSE", subtitle = "开发阶段LOBO：红色虚线为纯Theil--Sen基线；M2为增强模型中最佳，但仍略高", x = "特征组合", y = "开发阶段LOBO RMSE")
save_pub(f9, "fig9_ablation", 165, 105)

# 图10：九块测试电池的观测与未来预测。
obs_test <- cycle[cycle$battery_id %in% q3_summary$battery_id & cycle$cycle >= 100 & cycle$cycle <= 150, c("battery_id", "cycle", "SOH_sg", "policy")]
names(obs_test)[3] <- "SOH"; obs_test$type <- "已观测"
pred_test <- q3_pred[, c("battery_id", "cycle", "SOH_pred", "policy")]
names(pred_test)[3] <- "SOH"; pred_test$type <- "预测"
test_plot <- rbind(obs_test, pred_test)
test_plot$facet <- paste0("电池", test_plot$battery_id, "\n", short_policy(test_plot$policy))
f10 <- ggplot(test_plot, aes(cycle, SOH, colour = type)) +
  geom_vline(xintercept = 150, linewidth = .35, linetype = 3, colour = palette["grey"]) + geom_line(linewidth = .72) +
  facet_wrap(~facet, ncol = 3, scales = "free_y") +
  scale_colour_manual(values = c("已观测" = pc("dark"), "预测" = pc("blue"))) +
  labs(title = "九块测试电池151--200圈SOH预测", subtitle = "虚线为观测/预测边界；各分面纵轴按局部范围显示", x = "循环次数", y = "SOH", colour = NULL) +
  theme(legend.position = "bottom")
save_pub(f10, "fig10_test_predictions", 180, 150)

# 图11：完整寿命三均值只能提供间接压力测试，系统偏差不支持真实EOL准确性。
life_state_plot <- function(channel_name, title_text, unit_text, digits, colour_name) {
  d <- q3_life_state[q3_life_state$channel == channel_name, ]
  limits <- range(c(d$true_life_mean, d$predicted_life_mean))
  padding <- max(diff(limits) * 0.08, 10^(-digits))
  mae <- mean(abs(d$predicted_life_mean - d$true_life_mean))
  bias <- mean(d$predicted_life_mean - d$true_life_mean)
  ggplot(d, aes(true_life_mean, predicted_life_mean, label = battery_id)) +
    geom_abline(slope = 1, intercept = 0, linewidth = .55, linetype = 2, colour = pc("grey")) +
    geom_point(size = 2.2, colour = pc(colour_name), alpha = .88) +
    geom_text_repel(size = 2.5, family = font_cn, min.segment.length = 0,
                    box.padding = .25, point.padding = .2, max.overlaps = Inf) +
    coord_equal(xlim = limits + c(-padding, padding), ylim = limits + c(-padding, padding)) +
    labs(
      title = title_text,
      subtitle = sprintf(paste0("MAE= %.", digits, "f %s\n偏差= %.", digits, "f %s"), mae, unit_text, bias, unit_text),
      x = paste0("真实完整寿命均值（", unit_text, "）"),
      y = paste0("预测完整寿命均值（", unit_text, "）")
    )
}

f11_ir <- life_state_plot("IR", "内阻均值整体偏低", "Ω", 6, "blue")
f11_t <- life_state_plot("Tavg", "温度均值全部低估", "℃", 3, "red")
f11_c <- life_state_plot("ChargeTime", "充电时间均值整体低估", "min", 3, "orange")
f11 <- (f11_ir | f11_t | f11_c) +
  plot_annotation(
    title = "完整寿命三均值未能验证远期EOL",
    subtitle = "虚线为理想45°线；数字为电池编号。图中吻合度衡量状态均值，不代表EOL圈数准确率"
  ) &
  theme(plot.title = element_text(family = font_cn, face = "bold", size = 10),
        plot.subtitle = element_text(family = font_cn, colour = pc("grey"), size = 8, lineheight = 1.05))
save_pub(f11, "fig11_life_summary_validation", 183, 92)

# 图12：预测至200圈后再外推与观测至200圈后再外推的结构稳定性；不是EOL真值验证。
rho_eol <- cor(q3_eol_stability$life_predicted_trajectory, q3_eol_stability$life_observed_to_200_reference, method = "spearman")
med_rel <- median(q3_eol_stability$relative_difference_percent)
lim_eol <- range(c(q3_eol_stability$life_predicted_trajectory, q3_eol_stability$life_observed_to_200_reference))
f12 <- ggplot(q3_eol_stability, aes(life_observed_to_200_reference, life_predicted_trajectory)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, linewidth = .55, colour = pc("grey")) +
  geom_point(size = 2.0, alpha = .82, colour = pc("blue")) +
  scale_x_log10(labels = label_number(big.mark = ",")) +
  scale_y_log10(labels = label_number(big.mark = ",")) +
  coord_equal(xlim = lim_eol, ylim = lim_eol) +
  annotate("label", x = lim_eol[1] * 1.08, y = lim_eol[2] / 1.08,
           hjust = 0, vjust = 1, family = font_cn, size = 3.1,
           label = sprintf("中位相对差异 = %.1f%%\nSpearman ρ = %.3f", med_rel, rho_eol),
           fill = alpha("white", .9)) +
  labs(title = "短期SOH高精度不等于远期EOL稳定", subtitle = "比较两种中心化二次外推：预测151--200圈 vs 真实151--200圈\n两者都不是实际观测的EOL标签", x = "观察至200圈后的中心化二次EOL（圈）", y = "预测至200圈后的中心化二次EOL（圈）")
save_pub(f12, "fig12_eol_structure_audit", 165, 110)

# 图12b：部分池化主要压制少数极端EOL漂移。
pool_long <- rbind(
  data.frame(battery_id = q3_pool$battery_id, method = "个体EOL", error = 100 * q3_pool$individual_relative_error),
  data.frame(battery_id = q3_pool$battery_id, method = "0.75个体+0.25同策略", error = 100 * q3_pool$pooled_relative_error)
)
pool_long$method <- factor(pool_long$method, levels = c("个体EOL", "0.75个体+0.25同策略"))
f12b <- ggplot(pool_long, aes(method, error, group = battery_id)) +
  geom_line(colour = pc("lightgrey"), linewidth = .35, alpha = .7) +
  geom_point(aes(colour = method), size = 1.7, alpha = .78) +
  stat_summary(aes(group = method), fun = median, geom = "point", shape = 23, size = 3.8, fill = "white", colour = pc("dark")) +
  scale_colour_manual(values = c("个体EOL" = pc("grey"), "0.75个体+0.25同策略" = pc("teal")), guide = "none") +
  scale_y_continuous(labels = label_number(suffix = "%", accuracy = 1)) +
  labs(title = "同策略部分池化降低EOL截断漂移", subtitle = "40块完整训练电池的嵌套LOBO；菱形为中位数", x = NULL, y = "150圈与200圈EOL相对差")
save_pub(f12b, "fig12b_partial_pooling", 145, 95)

# 补充图：中心化二次只从候选稳定起点以后施加单调约束。
train_battery <- battery[battery$prediction_test == 0, ]
rate_targets <- as.numeric(quantile(train_battery$stable_rate, c(.1, .5, .9), na.rm = TRUE))
representative_ids <- unique(vapply(rate_targets, function(target) {
  train_battery$battery_id[which.min(abs(train_battery$stable_rate - target))]
}, numeric(1)))
fit_observed <- cycle[cycle$battery_id %in% representative_ids & cycle$cycle <= 200, c("battery_id", "cycle", "SOH_sg", "policy")]
fit_observed$facet <- paste0("电池", fit_observed$battery_id, "｜", short_policy(fit_observed$policy))

fit_spec <- q1_fits[
  q1_fits$battery_id %in% representative_ids &
    q1_fits$n0 == 31 & q1_fits$model %in% c("centered_quadratic", "power", "linear"),
]
fit_rows <- list(); fit_index <- 1
for (row_index in seq_len(nrow(fit_spec))) {
  row <- fit_spec[row_index, ]
  n_seq <- seq(row$n0, 200)
  prediction <- if (row$model == "linear") {
    row$a + row$b * n_seq
  } else if (row$model == "power") {
    row$a - row$b * n_seq^row$c
  } else {
    row$a - row$b * (n_seq - row$n0) - row$c * (n_seq - row$n0)^2
  }
  policy_name <- battery$policy[battery$battery_id == row$battery_id][1]
  fit_rows[[fit_index]] <- data.frame(
    battery_id = row$battery_id,
    cycle = n_seq,
    SOH = prediction,
    model = unname(c(linear = "线性", power = "幂律", centered_quadratic = "中心化二次")[row$model]),
    facet = paste0("电池", row$battery_id, "｜", short_policy(policy_name))
  )
  fit_index <- fit_index + 1
}
fit_curves <- do.call(rbind, fit_rows)
fit_curves$model <- factor(fit_curves$model, levels = c("中心化二次", "幂律", "线性"))
f_q1_fit <- ggplot() +
  geom_line(data = fit_observed, aes(cycle, SOH_sg), linewidth = .55, colour = pc("dark"), alpha = .75) +
  geom_vline(xintercept = 31, linewidth = .35, linetype = 3, colour = pc("teal")) +
  geom_line(data = fit_curves, aes(cycle, SOH, colour = model, linetype = model), linewidth = .82) +
  facet_wrap(~facet, ncol = 3, scales = "free_y") +
  scale_colour_manual(values = c("中心化二次" = pc("teal"), "幂律" = pc("blue"), "线性" = pc("orange"))) +
  scale_linetype_manual(values = c("中心化二次" = 1, "幂律" = 2, "线性" = 3)) +
  labs(
    title = "中心化二次从n₀=31以后约束单调加速退化",
    subtitle = "展示稳定退化速率10%、50%和90%分位附近的完整电池；黑线为观测SOH",
    x = "循环次数", y = "SOH", colour = NULL, linetype = NULL
  ) +
  theme(legend.position = "bottom")
save_pub(f_q1_fit, "fig4b_q1_fit_shapes", 180, 100)

# Q4图1：九种真实策略的经验Pareto前沿。
q4_existing$policy_label <- short_policy(q4_existing$policy)
q4_existing$pareto_observed <- tolower(as.character(q4_existing$pareto_observed)) == "true"
strategy_codes <- c(
  "3_6C-80PER_3_6C" = "S1", "80PER_3_6C" = "S2", "4_8C_80PER_4_8C" = "S3",
  "5C_67PER_4C_NEWSTRUCTURE" = "N1", "5_3C_54PER_4C_NEWSTRUCTURE" = "N2",
  "5_6C_19PER_4_6C_NEWSTRUCTURE" = "N3", "3_7C_31PER_5_9C_NEWSTRUCTURE" = "N4",
  "5_6C_36PER_4_3C_NEWSTRUCTURE" = "N5", "4_8C_80PER_4_8C_NEWSTRUCTURE" = "N6"
)
q4_existing$code <- unname(strategy_codes[q4_existing$policy])
q4_existing$r_plot <- q4_existing$r_stable * 1e4
q4_existing$r_q1_plot <- q4_existing$r_stable_q1 * 1e4
q4_existing$r_q3_plot <- q4_existing$r_stable_q3 * 1e4
q4_existing$status <- ifelse(q4_existing$pareto_observed, "经验Pareto", "被支配")
pareto_line <- q4_existing[q4_existing$pareto_observed, ]
pareto_line <- pareto_line[order(pareto_line$T_obs), ]
recommended_policy <- "5_3C_54PER_4C_NEWSTRUCTURE"
f_q4_pareto <- ggplot(q4_existing, aes(T_obs, r_plot)) +
  geom_segment(aes(x = T_obs_q1, xend = T_obs_q3, yend = r_plot), linewidth = .45, colour = pc("lightgrey")) +
  geom_segment(aes(y = r_q1_plot, yend = r_q3_plot, xend = T_obs), linewidth = .45, colour = pc("lightgrey")) +
  geom_path(data = pareto_line, linewidth = .65, colour = pc("teal"), linetype = 2) +
  geom_point(aes(fill = status, size = pareto_bootstrap_probability), shape = 21, colour = "white", stroke = .7) +
  geom_point(data = q4_existing[q4_existing$policy == recommended_policy, ], shape = 23, size = 4.1, fill = pc("orange"), colour = pc("dark"), stroke = .8) +
  geom_text_repel(aes(label = code), family = font_cn, fontface = "bold", size = 3.0,
                  min.segment.length = 0, box.padding = .3, point.padding = .25, max.overlaps = Inf) +
  scale_fill_manual(values = c("经验Pareto" = pc("teal"), "被支配" = pc("grey"))) +
  scale_size_continuous(range = c(2.2, 4.2), labels = label_percent(accuracy = 1)) +
  labs(
    title = "九种真实策略形成三点经验Pareto前沿",
    subtitle = "横纵误差线为策略内四分位区间；策略代码对应表中名称，橙色菱形为最终推荐N2",
    x = "观测0–80%快充时间中位数（min）", y = expression(10^4 %.% r^stable),
    fill = NULL, size = "Bootstrap非支配频率"
  ) +
  theme(legend.position = "bottom")
save_pub(f_q4_pareto, "fig13_q4_empirical_pareto", 180, 112)

# Q4图2：双重凸包内的A-D局部候选及鲁棒目标。
q4_grid$q90_plot <- q4_grid$selection_q90 * 1e4
observed_q4 <- q4_compare[q4_compare$design == "best_existing", ]
candidate_q4 <- q4_compare[q4_compare$design == "optimized_local_candidate", ]
new_design <- read_utf8("问题2", "q2_04_策略参数与SOC暴露.csv")
new_design <- new_design[new_design$dataset_id == 3, ]
new_design$label <- unname(strategy_codes[new_design$policy])
hull_index <- chull(new_design$A, new_design$D_50)
hull_ad <- new_design[c(hull_index, hull_index[1]), ]
f_q4_local <- ggplot(q4_grid, aes(A, D_50)) +
  geom_point(aes(colour = q90_plot), size = 1.25, alpha = .72) +
  geom_path(data = hull_ad, aes(A, D_50), inherit.aes = FALSE, linewidth = .55, colour = pc("dark"), linetype = 2) +
  geom_point(data = new_design, aes(A, D_50), inherit.aes = FALSE, shape = 21, size = 3.0, fill = "white", colour = pc("dark"), stroke = .8) +
  geom_text_repel(data = new_design, aes(A, D_50, label = label), inherit.aes = FALSE,
                  family = font_cn, fontface = "bold", size = 2.9,
                  min.segment.length = 0, box.padding = .25, point.padding = .2, max.overlaps = Inf) +
  geom_point(data = observed_q4, aes(A, D_50), inherit.aes = FALSE, shape = 23, size = 4.2, fill = pc("orange"), colour = pc("dark"), stroke = .8) +
  geom_point(data = candidate_q4, aes(A, D_50), inherit.aes = FALSE, shape = 8, size = 4.0, colour = pc("red"), stroke = 1.0) +
  geom_text_repel(data = candidate_q4, aes(A, D_50, label = "C*"), inherit.aes = FALSE,
                  family = font_cn, fontface = "bold", size = 3.0, colour = pc("red"),
                  min.segment.length = 0, box.padding = .25, point.padding = .3) +
  scale_colour_gradientn(colours = c(pc("teal"), pc("lightblue"), pc("orange"), pc("red"))) +
  labs(
    title = "局部候选同时受原始参数域与A–D特征域约束",
    subtitle = "颜色为选择Bootstrap的退化预测90%分位；橙菱形为N2，红星为待验证候选C*",
    x = "总体倍率 A", y = "中高SOC与低SOC暴露差 D", colour = expression(Q[0.90](10^4 %.% hat(r)^stable))
  ) +
  theme(legend.position = "right")
save_pub(f_q4_local, "fig14_q4_local_robust_search", 180, 112)

# Q4图3：独立验证Bootstrap显示局部候选没有稳定优于已有策略。
q4_paired$delta_plot <- q4_paired$candidate_minus_existing * 1e5
prob_better <- mean(q4_paired$candidate_minus_existing < 0)
delta_q90 <- quantile(q4_paired$delta_plot, .9)
lopo_mae_plot <- 7.533663618231768e-06 * 1e5
f_q4_boot <- ggplot(q4_paired, aes(delta_plot)) +
  geom_density(fill = alpha(pc("lightblue"), .8), colour = pc("blue"), linewidth = .8, adjust = 1.05) +
  geom_vline(xintercept = 0, colour = pc("red"), linewidth = .7) +
  geom_vline(xintercept = c(-lopo_mae_plot, lopo_mae_plot), colour = pc("grey"), linewidth = .5, linetype = 3) +
  annotate("label", x = Inf, y = Inf, hjust = 1.05, vjust = 1.15, family = font_cn, size = 3.0,
           label = sprintf("P(候选更优)=%.1f%%\n配对差Q90=%.3f×10⁻⁵\n灰线=±LOPO MAE", 100 * prob_better, delta_q90),
           fill = alpha("white", .92)) +
  labs(
    title = "局部候选未形成可分辨的稳定改进",
    subtitle = "独立的5000次策略内Bootstrap；横轴为候选减去最佳已有策略的预测退化速率",
    x = expression(10^5 %.% (hat(r)[candidate]^stable - hat(r)[existing]^stable)), y = "密度"
  )
save_pub(f_q4_boot, "fig15_q4_bootstrap_difference", 165, 100)

cat("Figures written to:", fig_dir, "\n")
