from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:

    dim =  512
    cosine_threshold = 0.3

  #  dim = 256
  #  cosine_threshold = 0.45  # 相似度阈

   # dim = 128
   # cosine_threshold = 0.6  # 相似度阈

   # dim = 64
   # cosine_threshold = 0.7  # 相似度阈

    file_path = f"/data2/zxc_data/foldseek_database/foldseek_database/sp_cif_file/gitdata/SSAlign/svd{dim}_cos_greater_{cosine_threshold}.csv"

    # 数据加载
    df = pd.read_csv(file_path)
    if not {'Cosine_Similarity', 'Avg_TM_Score'}.issubset(df.columns):
        raise ValueError("CSV文件中缺少必要列")

    x = df['Cosine_Similarity'].values
    y = df['Avg_TM_Score'].values

    # 线性拟合
    coefficients = np.polyfit(x, y, deg=1)
    slope, intercept = coefficients
    linear_func = np.poly1d(coefficients)


    # 计算皮尔逊相关系数及p值
    corr, p_value = pearsonr(x, y)

    print(f"Pearson Correlation Coefficient: {corr:.4f}")
    print(f"P-value: {p_value:.4g}")  # p值用于检验相关性是否显著

    # 输出结果
    print(f"拟合结果:\n斜率 = {slope:.4f}\n截距 = {intercept:.4f}\n")

except FileNotFoundError:
    print("错误：未找到CSV文件")
except Exception as e:
    print(f"发生错误: {str(e)}")






