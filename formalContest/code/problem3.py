"""
利用指标差异解释问题,用机器学习模型分析比赛数据,识别可能的转折点指标(分数差,发球方,连续得分,破发点等)
特征工程:对data对象进行属性提取,选择可能对比赛势头的转换有影响的特征
模型选择:梯度提升树
训练与验证:直接用7000条数据进行验证
预测目标:预测目标是一个二元变量,如果比赛流向在接下来几个球中转向另一位选手,则Y=1,否则Y=0.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = '..\\data\\R3M1.csv'
data = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升树
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns

"""
选用的特征:
    得分差:当前比分差异
    发球权:当前发球方,发球方拥有更高的得分概率
    连续得分数:选手连续得分的数量
    重要得分:是否是破发点
    
"""
# 初始化势头属性
data['momentum_score'] = 0
data['momentum_score_1'] = 0
data['momentum_score_2'] = 0

# 初始化算法中连续得分权重变量
consecutive_wins_1 = 0
consecutive_wins_2 = 0

# 破发后连续得分权重变量
score_after_break_1 = 0
score_after_break_2 = 0

# 初始化算法中1号和2号的势头变量
momentum_score = 0
momentum_score_1 = 0
momentum_score_2 = 0

# 初始化分数差属性为0
data['score_differential'] = 0
# 在df对象中加入连续得分数属性,用于写入df对象
data['consecutive'] = 0
data['critical_point'] = 0
data['consecutive_1'] = 0
data['consecutive_2'] = 0
# 初始化连续得分开启变量,用于程序逻辑
crit = 0
consecutive_1 = 0
consecutive_2 = 0

p1_points_won = 0
p2_points_won = 0
score_differential = 0

for index, row in data.iterrows():
    # 计算分差 #1-#2
    score_differential = row['p1_points_won'] - row['p2_points_won']
    row['score_differential'] = score_differential
    data.at[index, 'score_differential'] = row['score_differential']

    # 计算连续得分数
    if row['point_victor'] == 1:

        S_t_1 = 1.2 if row['server'] == 1 else 1.0
        S_t_2 = 1.2 if row['server'] == 2 else 1.0

        # 1号选手的连续得分变量变为1
        consecutive_wins_1 += 1
        consecutive_1 += 1
        score_after_break_1 += 1 if row['p1_break_pt_won'] == 1 else 0
        # 2号选手的连续得分变量变为0
        consecutive_2 = 0
        consecutive_wins_2 = 0
        score_after_break_2 = 0

        row['consecutive_1'] = consecutive_1
        row['consecutive_2'] = consecutive_2

        if row['server'] == 2:
            row['critical_point'] = 1
        else:
            row['critical_point'] = 0

        C_t_1 = 2 + score_after_break_1 * 0.4 if row['p1_break_pt_won'] == 1 else 1 + consecutive_wins_1 * 0.2
        C_t_2 = 2 + score_after_break_2 * 0.4 if row['p2_break_pt_won'] == 1 else 1 + consecutive_wins_2 * 0.2
        momentum_score_1 += (S_t_1 * C_t_1)
        momentum_score_2 += (S_t_2 * C_t_2)

        data.at[index, 'critical_point'] = row['critical_point']

    elif row['point_victor'] == 2:

        # 如果发球得分,那么发球权重为1.2,否则1.0
        S_t_1 = 1.2 if row['server'] == 1 else 1.0
        S_t_2 = 1.2 if row['server'] == 2 else 1.0
        # 2号选手的连续得分变量变为1
        consecutive_2 += 1
        consecutive_wins_2 += 1
        score_after_break_2 += 1 if row['p2_break_pt_won'] == 1 else 0
        # 1号选手的连续得分变量变为0
        consecutive_1 = 0
        consecutive_wins_1 = 0
        score_after_break_1 = 0

        C_t_1 = 2 + score_after_break_1 * 0.4 if row['p1_break_pt_won'] == 1 else 1 + consecutive_wins_1 * 0.2
        C_t_2 = 2 + score_after_break_2 * 0.4 if row['p2_break_pt_won'] == 1 else 1 + consecutive_wins_2 * 0.2
        momentum_score_1 += (S_t_1 * C_t_1)
        momentum_score_2 += (S_t_2 * C_t_2)

        row['consecutive_1'] = consecutive_1
        row['consecutive_2'] = consecutive_2
        if row['server'] == 1:
            row['critical_point'] = 1
        else:
            row['critical_point'] = 0
        data.at[index, 'critical_point'] = row['critical_point']

    momentum_score = (momentum_score_1 - momentum_score_2)
    data.at[index, 'momentum_score'] = momentum_score
    data.at[index, 'momentum_score_1'] = momentum_score_1
    data.at[index, 'momentum_score_2'] = momentum_score_2

    if consecutive_1 == 0:
        row['consecutive'] = consecutive_2
    elif consecutive_2 == 0:
        row['consecutive'] = consecutive_1
    else:
        row['consecutive'] = 0

    data.at[index, 'consecutive_1'] = row['consecutive_1']
    data.at[index, 'consecutive_2'] = row['consecutive_2']
    data.at[index, 'consecutive'] = row['consecutive']

# print('连续得分情况')
# for line in data['consecutive'][0:10]:
#     print(line)
# print("破发点情况")
# for line in data['critical_point'][0:10]:
#     print(line)
# print('一号连续得分情况')
# for line in data['consecutive_1'][0:10]:
#     print(line)
# print('二号连续得分情况')
# for line in data['consecutive_2'][0:10]:
#     print(line)
# print('总体连续得分情况')
# for line in data['consecutive'][0:10]:
#     print(line)

# 梯度提升树
features = data[['score_differential', 'server', 'consecutive', 'critical_point']]
target = data['momentum_score']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Model Accuracy: {accuracy}")

# # 网格搜索超参数调优,这段代码可以找出最好的参数,对R3M1,学习率0.1 深度4 n_estimator 100时最好
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 4, 5]
# }
# model = GradientBoostingRegressor(random_state=42)
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# print(f'最好的参数: {best_params}')
# best_model = grid_search.best_estimator_
# prediction_best = best_model.predict(X_test)

# mse方差
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# RMSE均方根误差
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# 决定系数(目标对变量变化的解释能力,越接近1拟合越好)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

# 特征重要性图:哪些特征对预测的贡献最大
feature_importance = model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=features.columns)
plt.title('Feature Importance')
plt.xlabel('importance')
plt.ylabel('features')
plt.title('Feature Importance in R3M1')
plt.savefig('..\\image\\Q3\\R3M1_feature_Importance.jpg')
plt.show()

