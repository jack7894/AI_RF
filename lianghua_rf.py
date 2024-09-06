import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('tending_data.csv')  # 假设您的数据存储在CSV文件中

# 2. 特征工程
df['SMA_10'] = df['close'].rolling(window=10).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()


df.dropna(inplace=True)

# 3. 准备特征和目标变量
features = ['open', 'high', 'low', 'close', 'SMA_10', 'SMA_30']
X = df[features]
y = np.where(df['close'] > df['close'].shift(-1), 1, 0)  # 1表示价格上涨,0表示下跌

# 4. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 5. 创建和训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 6. 在测试集上进行预测
y_pred = rf_model.predict(X_test)

# 7. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 8. 特征重要性
feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
print("\n特征重要性:")
print(feature_importance.sort_values('importance', ascending=False))



# 找到特征重要性最高的树
feature_importances = [tree.feature_importances_ for tree in rf_model.estimators_]
best_tree_index = np.argmax([np.corrcoef(imp, rf_model.feature_importances_)[0, 1] for imp in feature_importances])
best_tree = rf_model.estimators_[best_tree_index]

# 使用这棵树来绘图
plt.figure(figsize=(40,20))
plot_tree(best_tree, 
          feature_names=features,  
          class_names=['下跌', '上涨'],
          filled=True, 
          rounded=True,  
          fontsize=12,
          max_depth=5,
          proportion=True,
          precision=2)
plt.savefig('best_representative_tree.png', dpi=300, bbox_inches='tight')
plt.close()
