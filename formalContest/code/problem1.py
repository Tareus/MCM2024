"""
第一问需要建立时间序列模型,状态空间模型,关于时间点计算当下比赛势头的量化值,并考虑球员发球权的影响.
Y_t为时间点t的比赛状态,X_t为影响比赛状态的因素集合,包括当前局分,发球权等等.
状态空间模型可以表示为:$ Y_t = f(X_t,\sita) + \epsilon_t $
$ M_t = \alpha\sum_{i=1}^{t}P_i-\beta\sum_{i=1}^{t}P_i $
epsilon_t是随机误差项,可能与各种场外因素或者队员心理素质有关

"""
import pandas as pd
import matplotlib.pyplot as plt

file_path = "..\\data\\R7M1.csv"
df = pd.read_csv(file_path)

# DataFrame的所有属性
print("DataFrame的信息:")
print(df.info())

# DataFrame 的头部（前几行数据）：
print("\nDataFrame的头部:")
print(df.head())

# DataFrame 的统计摘要：
print("\nDataFrame的统计摘要:")
print(df.describe())

# DataFrame 的列名：
print("\nDataFrame的列名:")
print(df.columns)


df['momentum_score'] = 0
momentum_score = 0
consecutive_wins = 0
for index, row in df.iterrows():
    """
    M_t为当前时间点t的势头分数,P_t为当前时间点是否得分,S_t为发球权重(发球方为1.2,接球方为1.0),C_t为连续得分加成0.2(第一次得分为1,之后每连续得一分加0.2)
    M_t = M_(t-1)+(P_t*S_t*C_t),如果当前时间点的这一球失分,那么势头不变,但是连续得分加成清零
    """
    P_t = 1 if row['point_victor'] == row['server'] else 0  # 发球得分
    S_t = 1.2 if row['server'] == 1 else 1.0  # Novak Djokovic是发球方
    if P_t == 1:
        consecutive_wins += 1
    else:
        consecutive_wins = 0
    C_t = 1 + consecutive_wins * 0.2
    momentum_score += (P_t * S_t * C_t)
    df.at[index, 'momentum_score'] = momentum_score
plt.plot(df['point_no'], df['momentum_score'], label='Momentum Score')
plt.xlabel('Point Number')
plt.ylabel('Moment Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.show()
