# 导入必要的库
import pandas as pd

# 读取数据
titanic_data = pd.read_csv(r"C:\Users\Forword Sun\Desktop\titanic\train.csv")

# 查看数据的基本信息
print(titanic_data.info())

# 查看数据的统计摘要
print(titanic_data.describe())

# 查看前几行数据
print(titanic_data.head())


