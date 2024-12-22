# 載入必要的套件
library(readxl)
library(dplyr)
library(ggplot2)
library(writexl)

# 讀取Excel檔案
lenet_results_small <- read_excel("C:/Users/88693/Downloads/transfer/lenet_results_small.xlsx")
lenet_results_big <- read_excel("C:/Users/88693/Downloads/transfer/lenet_results_big.xlsx")
model_results_small <- read_excel("C:/Users/88693/Downloads/transfer/model_results_small.xlsx")
model_results_big <- read_excel("C:/Users/88693/Downloads/transfer/model_results_big.xlsx")

# 合併資料
lenet_results_small <- lenet_results_small %>% mutate(Model = "LeNet_Small")
lenet_results_big <- lenet_results_big %>% mutate(Model = "LeNet_Big")
model_results_small <- model_results_small %>% mutate(Model = "Squeeze_Small")
model_results_big <- model_results_big %>% mutate(Model = "Squeeze_Big")

all_data <- bind_rows(lenet_results_small, lenet_results_big, model_results_small, model_results_big)

# 計算平均值、標準差和95%信賴區間
stat_summary <- all_data %>%
  group_by(Epoch, Model) %>%
  summarize(
    Accuracy_Mean = mean(Accuracy, na.rm = TRUE),
    Accuracy_SD = sd(Accuracy, na.rm = TRUE),
    Accuracy_CI_Lower = Accuracy_Mean - qt(0.975, df = n() - 1) * (Accuracy_SD / sqrt(n())),
    Accuracy_CI_Upper = Accuracy_Mean + qt(0.975, df = n() - 1) * (Accuracy_SD / sqrt(n())),
    Precision_Mean = mean(Precision, na.rm = TRUE),
    Precision_SD = sd(Precision, na.rm = TRUE),
    Precision_CI_Lower = Precision_Mean - qt(0.975, df = n() - 1) * (Precision_SD / sqrt(n())),
    Precision_CI_Upper = Precision_Mean + qt(0.975, df = n() - 1) * (Precision_SD / sqrt(n())),
    Recall_Mean = mean(Recall, na.rm = TRUE),
    Recall_SD = sd(Recall, na.rm = TRUE),
    Recall_CI_Lower = Recall_Mean - qt(0.975, df = n() - 1) * (Recall_SD / sqrt(n())),
    Recall_CI_Upper = Recall_Mean + qt(0.975, df = n() - 1) * (Recall_SD / sqrt(n())),
    F1_Score_Mean = mean(F1_Score, na.rm = TRUE),
    F1_Score_SD = sd(F1_Score, na.rm = TRUE),
    F1_Score_CI_Lower = F1_Score_Mean - qt(0.975, df = n() - 1) * (F1_Score_SD / sqrt(n())),
    F1_Score_CI_Upper = F1_Score_Mean + qt(0.975, df = n() - 1) * (F1_Score_SD / sqrt(n())),
    ROC_AUC_Mean = mean(ROC_AUC, na.rm = TRUE),
    ROC_AUC_SD = sd(ROC_AUC, na.rm = TRUE),
    ROC_AUC_CI_Lower = ROC_AUC_Mean - qt(0.975, df = n() - 1) * (ROC_AUC_SD / sqrt(n())),
    ROC_AUC_CI_Upper = ROC_AUC_Mean + qt(0.975, df = n() - 1) * (ROC_AUC_SD / sqrt(n()))
  )

# 查看統計摘要
print(stat_summary)

# 定義繪圖函數
plot_metric <- function(metric_name, y_label) {
  ggplot(stat_summary, aes(x = Epoch, y = !!sym(paste0(metric_name, "_Mean")), color = Model)) +
    geom_line(size = 1) +
    geom_errorbar(aes(
      ymin = !!sym(paste0(metric_name, "_Mean")) - !!sym(paste0(metric_name, "_SD")),
      ymax = !!sym(paste0(metric_name, "_Mean")) + !!sym(paste0(metric_name, "_SD"))
    ), width = 0.2, alpha = 0.5) +
    labs(title = paste(metric_name, "Over Epochs"), x = "Epoch", y = y_label, color = "Model") +
    theme_minimal()
}

# 繪製圖表
accuracy_plot <- plot_metric("Accuracy", "Accuracy")
precision_plot <- plot_metric("Precision", "Precision")
recall_plot <- plot_metric("Recall", "Recall")
f1score_plot <- plot_metric("F1_Score", "F1 Score")
roc_auc_plot <- plot_metric("ROC_AUC", "ROC AUC")

# 顯示圖表
print(accuracy_plot)
print(precision_plot)
print(recall_plot)
print(f1score_plot)
print(roc_auc_plot)

# 匯出統計摘要為Excel檔案
write_xlsx(stat_summary, "C:/Users/88693/Downloads/transfer/stat_summary.xlsx")
cat("統計摘要已匯出至 C:/Users/88693/Downloads/transfer/stat_summary.xlsx\n")
