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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升树
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns
import time

file_path = '..\\data\\R3M1_new.csv'
data = pd.read_csv(file_path)
"""
选用的特征:所有特征
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
data['elapsed_time_second'] = 0
# 初始化连续得分开启变量,用于程序逻辑
crit = 0
consecutive_1 = 0
consecutive_2 = 0

p1_points_won = 0
p2_points_won = 0
score_differential = 0

# 将时间转换为秒
data['elapsed_time'] = pd.to_datetime(data['elapsed_time'])
data['elapsed_time_second'] = data['elapsed_time'].apply(lambda x: int(x.timestamp()))
print(data)

# 数值映射
mapping_1 = {'F': 1, 'B': 2, '': 3}
mapping_2 = {'B': 1, 'BC': 2, 'BW': 3, 'C': 4, 'W': 5, '': 6}
mapping_3 = {'CLT': 1, 'NCLT': 0, '': 2}
mapping_4 = {'D': 1, 'ND': 0, '': 3}
data['winner_shot_type'] = data['winner_shot_type'].map(mapping_1)
data['serve_width'] = data['serve_width'].map(mapping_2)
data['serve_depth'] = data['serve_depth'].map(mapping_3)
data['return_depth'] = data['return_depth'].map(mapping_4)
# 填充空缺值
data.fillna(value=0, inplace=True)
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
features = data[[
    'score_differential',  # 分数差
    'server',  # 发球方
    'consecutive',  # 连续得分
    'critical_point',  # 是否为关键点
    'elapsed_time_second',  # 比赛进行的时间
    'set_no',
    'game_no',
    'point_no',
    'p1_sets',
    'p2_sets',
    'p1_games',
    'p2_games',
    # p1_score 和 p2_score因为有AD,直接用12345代替
    'serve_no',
    'point_victor',
    'p1_points_won',
    'p2_points_won',
    'game_victor',
    'set_victor',
    'p1_ace',
    'p2_ace',
    'p1_winner',
    'p2_winner',
    'winner_shot_type',
    'p1_double_fault',
    'p2_double_fault',
    'p1_unf_err',
    'p2_unf_err',
    'p1_net_pt',
    'p2_net_pt',
    'p1_net_pt_won',
    'p2_net_pt_won',
    'p1_break_pt',
    'p2_break_pt',
    'p1_break_pt_won',
    'p2_break_pt_won',
    'p1_break_pt_missed',
    'p2_break_pt_missed',
    'p1_distance_run',
    'p2_distance_run',
    'rally_count',
    'speed_mph',
    'serve_width',
    'serve_depth',
    'return_depth'
]]
target = data['momentum_score']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Model Accuracy: {accuracy}")

# # 网格搜索超参数调优,这段代码可以找出最好的参数,对R3M1,学习率0.1 深度4 n_estimator 100时最好
# 所有属性都用上,深度3最好
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
plt.savefig('..\\image\\Q3\\R3M1_all_feature_importance.jpg')
plt.show()

from pdpbox import info_plots, get_dataset, pdp
X = features
interact_obj = pdp.pdp_interact(model=model, dataset=X, model_features=X.columns, features=['score_differential',
                                                                                            'critical_point',
                                                                                            'consecutive',
                                                                                            'server',
                                                                                            'elapsed_time_second'])
pdp.pdp_interact_plot(interact_obj, ['consecutive', 'elapsed_time_second'])
plt.show()
