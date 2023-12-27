import pandas as pd
from sklearn.preprocessing import StandardScaler
# 读取数据
titanic_data = pd.read_csv(r"C:\Users\Forword Sun\Desktop\titanic\train.csv")
# 创建一个新的 DataFrame，以保持原始数据不变
processed_train = titanic_data.copy()
# 处理缺失值
mean_age = processed_train['Age'].mean()
processed_train['Age'] = processed_train['Age'].fillna(mean_age)
processed_train['Embarked'] = processed_train['Embarked'].fillna('S')

# 将 'Name' 列中的名字转换为昵称（使用名字的第二个单词）
processed_train['nickname'] = processed_train['Name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else x)

# 删除不需要的列
processed_train = processed_train.drop(columns=['Name', 'Ticket', 'Cabin'])

# 对分类特征进行独热编码
categorical_features = ['Sex', 'Embarked']
processed_train = pd.get_dummies(processed_train, columns=categorical_features)

# 对连续特征进行归一化处理
continuous_features = ['Fare']
scaler = StandardScaler()
processed_train[continuous_features] = scaler.fit_transform(processed_train[continuous_features])
# 显示修改后的数据的前几行以确认更改
print(processed_train.head())
# 保存修改后的DataFrame到新的CSV文件
processed_train.to_csv(r"C:\Users\Forword Sun\Desktop\titanic\processed_train.csv", index=False)
