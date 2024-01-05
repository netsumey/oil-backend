import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data import Operator

# %%
# 特征筛选
train_path = "./dataset/yali_train_missing.csv"

frames = pd.read_csv(train_path)
frames.dropna(inplace=True)

correlation_matrix = frames.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
# 全选

# %%
# 填充
operator = Operator(path=train_path, out_path="")
operator.handle_missing_values(method="Mean")

# %%
# 归一化
operator.normalize_columns(method="Z")

# %%
# ...

# %%
