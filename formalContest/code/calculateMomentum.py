import pandas as pd
import matplotlib.pyplot as plt

"""
分别计算当前时间点两个球员的势头量化值,再作差,从而得出整个比赛的势头走向
"""
# 文件路径
image_path = '..\\image\\MomentumScore\\R3M1B.jpg'
file_path = '..\\data\\R3M1.csv'
image_path_1 = '..\\image\\MomentumScore\\R3M1B_1.jpg'
image_path_2 = '..\\image\\MomentumScore\\R3M1B_2.jpg'
# 读取文件作为DataFrame对象data
data = pd.read_csv(file_path)

# 初始化势头属性
data['momentum_score'] = 0
data['momentum_score_1'] = 0
data['momentum_score_2'] = 0

# 初始化算法中1号和2号的势头变量
momentum_score = 0
momentum_score_1 = 0
momentum_score_2 = 0

# 初始化算法中连续得分权重变量
consecutive_wins_1 = 0
consecutive_wins_2 = 0

# 破发点权重初始化
break_1 = 1
break_2 = 1

# 破发后连续得分权重变量
score_after_break_1 = 0
score_after_break_2 = 0

# 以1号选手得分为正,2号选手得分为负
for index, row in data.iterrows():

    # 如果当前1号选手得分
    if row['point_victor'] == 1:

        # 如果发球得分,那么发球权重为1.2,否则1.0
        S_t_1 = 1.2 if row['server'] == 1 else 1.0
        S_t_2 = 1.2 if row['server'] == 2 else 1.0


        # 1号选手连续得分加成基础值变为1(能获得这项加成)
        consecutive_wins_1 += 1
        # 如果是破发,破发后连续得分基础值变为1(能获得这项加成)
        score_after_break_1 += 1 if row['p1_break_pt_won'] == 1 else 0
        # 1号终结了2号的连杀,2号连续得分基础值清零,移除buff
        consecutive_wins_2 = 0
        # 2号的破发连续得分加成清零
        score_after_break_2 = 0
        # 连续得分权重更新

        C_t_1 = 2 + score_after_break_1 * 0.4 if row['p1_break_pt_won'] == 1 else 1 + consecutive_wins_1 * 0.2
        C_t_2 = 2 + score_after_break_2 * 0.4 if row['p2_break_pt_won'] == 1 else 1 + consecutive_wins_2 * 0.2
        momentum_score_1 += (S_t_1 * C_t_1)
        momentum_score_2 += (S_t_2 * C_t_2)

        # 如果这是一个1号选手的破发点,那么需要在图中将这个破发点标出来

    # 如果当前2号选手得分
    elif row['point_victor'] == 2:

        # 如果发球得分,那么发球权重为1.2,否则1.0
        S_t_1 = 1.2 if row['server'] == 1 else 1.0
        S_t_2 = 1.2 if row['server'] == 2 else 1.0

        # 2号选手连续得分数加一(决定是否启动加成)
        consecutive_wins_2 += 1
        score_after_break_2 += 1 if row['p2_break_pt_won'] == 1 else 0
        # 2号终结了1号的连杀
        consecutive_wins_1 = 0
        score_after_break_1 = 0

        C_t_1 = 2 + score_after_break_1 * 0.4 if row['p1_break_pt_won'] == 1 else 1 + consecutive_wins_1 * 0.2
        C_t_2 = 2 + score_after_break_2 * 0.4 if row['p2_break_pt_won'] == 1 else 1 + consecutive_wins_2 * 0.2
        momentum_score_1 += (S_t_1 * C_t_1)
        momentum_score_2 += (S_t_2 * C_t_2)

    # 这里对应两个ΣMt相减的公式
    momentum_score = (momentum_score_1 - momentum_score_2)
    data.at[index, 'momentum_score'] = momentum_score
    data.at[index, 'momentum_score_1'] = momentum_score_1
    data.at[index, 'momentum_score_2'] = momentum_score_2

# 可视化绘图
plt.plot(data['point_no'], data['momentum_score'], label='Momentum Score')
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.savefig(image_path)
plt.show()

plt.plot(data['point_no'], data['momentum_score_1'], label='Momentum Score')
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.savefig(image_path_1)
plt.show()

plt.plot(data['point_no'], data['momentum_score_2'], label='Momentum Score')
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.savefig(image_path_2)
plt.show()
