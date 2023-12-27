import pandas as pd

# 读取数据
titanic_data = pd.read_csv(r"C:\Users\Forword Sun\Desktop\titanic\processed_train.csv")
# 移除 'nickname' 列并计算相关系数矩阵
correlation_matrix = titanic_data.drop('nickname', axis=1).corr()

# 查看 'Survived' 列与其他特征之间的相关性
survival_correlation = correlation_matrix['Survived']

# 打印相关性
print(survival_correlation)
