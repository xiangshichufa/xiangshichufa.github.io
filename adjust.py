import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 读取数据
titanic_data = pd.read_csv(r"C:\Users\Forword Sun\Desktop\titanic\processed_train.csv")

# 选择特征和标签
selected_features = ['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch','Fare']
X = titanic_data[selected_features]
y = titanic_data['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化模型
# 默认参数模型
svm_default = SVC(probability=True)
knn_default = KNeighborsClassifier()

# 调整参数的模型
svm_adjusted = SVC(C=1.16, probability=True)  # 增加C的值
knn_adjusted = KNeighborsClassifier(n_neighbors=40)  # 增加邻居数

# 训练和评估函数
def train_evaluate(model):
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    predictions = (prob > 0.5).astype(int)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prob)
    return precision, recall, f1, roc_auc

# 训练和评估默认参数模型
svm_default_metrics = train_evaluate(svm_default)
knn_default_metrics = train_evaluate(knn_default)

# 训练和评估调整参数模型
svm_adjusted_metrics = train_evaluate(svm_adjusted)
knn_adjusted_metrics = train_evaluate(knn_adjusted)

# 打印评估结果
def print_metrics(metrics, model_name):
    print(f"{model_name} Performance:")
    print(f"Precision: {metrics[0]}, Recall: {metrics[1]}, F1-Score: {metrics[2]}, AUC-ROC: {metrics[3]}\n")

print_metrics(svm_default_metrics, "SVM Default")
print_metrics(knn_default_metrics, "KNN Default")
print_metrics(svm_adjusted_metrics, "SVM Adjusted")
print_metrics(knn_adjusted_metrics, "KNN Adjusted")
