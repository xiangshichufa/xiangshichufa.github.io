import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 读取数据
titanic_data = pd.read_csv(r"C:\Users\Forword Sun\Desktop\titanic\processed_train.csv")

# 选择特征和标签
selected_features = ['Pclass', 'Age',  'SibSp', 'Parch','Sex_female','Sex_male']  # 确保所有特征都是数值型
X = titanic_data[selected_features]
y = titanic_data['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM模型
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_prob = svm_model.predict_proba(X_test)[:, 1]

# KNN模型
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_prob = knn_model.predict_proba(X_test)[:, 1]

# 计算评估指标
def evaluate_model(probabilities, y_test):
    predictions = (probabilities > 0.5).astype(int)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    return precision, recall, f1, roc_auc

# 评估SVM模型
svm_precision, svm_recall, svm_f1, svm_roc_auc = evaluate_model(svm_prob, y_test)

# 评估KNN模型
knn_precision, knn_recall, knn_f1, knn_roc_auc = evaluate_model(knn_prob, y_test)

# 打印评估结果
print("SVM Performance:\n")
print(f"Precision: {svm_precision}\nRecall: {svm_recall}\nF1-Score: {svm_f1}\nAUC-ROC: {svm_roc_auc}\n")

print("KNN Performance:\n")
print(f"Precision: {knn_precision}\nRecall: {knn_recall}\nF1-Score: {knn_f1}\nAUC-ROC: {knn_roc_auc}\n")
