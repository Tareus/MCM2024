import pandas as pd
import matplotlib.pyplot as plt


class TennisMatch:
    def __init__(self):
        self.sets = 0
        self.games = 0
        self.current_score = "0-0"
        self.is_serving = False
        self.completion = 0.0
        self.game_score = 0.0  # 记录每一局的比分


class TennisMatchData:
    def __init__(self):
        self.playerA = TennisMatch()
        self.playerB = TennisMatch()

    def read_data_from_excel(self, file_path):
        df = pd.read_csv(file_path)
        for i in range(len(df)):
            # 更新比分
            self.playerA.sets = df.iloc[i, 7]
            self.playerA.games = df.iloc[i, 9]
            self.playerA.current_score = df.iloc[i, 11]
            self.playerA.is_serving = df.iloc[i, 13] == '1'

            self.playerB.sets = df.iloc[i, 8]
            self.playerB.games = df.iloc[i, 10]
            self.playerB.current_score = df.iloc[i, 12]
            self.playerB.is_serving = df.iloc[i, 13] == '2'

            # 更新每一局的比分
            score_map = {"15": 0.01, "30": 0.02, "40": 0.03, "AD": 0.04}
            self.playerA.game_score = score_map.get(self.playerA.current_score.split("-")[0], 0) + (
                0.01 if self.playerA.is_serving else 0)
            self.playerB.game_score = score_map.get(self.playerB.current_score.split("-")[1], 0) + (
                0.01 if self.playerB.is_serving else 0)

            # 更新比赛完成度
            self.playerA.completion += self.playerA.game_score + (
                1 / 3 if self.playerA.sets > df.iloc[i - 1, 0] else 0) + (
                                           0.05 if self.playerA.games > df.iloc[i - 1, 1] else 0)
            self.playerB.completion += self.playerB.game_score + (
                1 / 3 if self.playerB.sets > df.iloc[i - 1, 4] else 0) + (
                                           0.05 if self.playerB.games > df.iloc[i - 1, 5] else 0)

    def plot_completion(self):
        plt.plot([player.completion for player in [self.playerA, self.playerB]])
        plt.xlabel('Time')
        plt.ylabel('Completion')
        plt.title('Player Completion Over Time')
        plt.legend(['Player A', 'Player B'])
        plt.show()

        # 展示折线图之后，减去每一局的比分
        self.playerA.completion -= self.playerA.game_score
        self.playerB.completion -= self.playerB.game_score

    def calculate_win_rate(self):
        # 计算每个选手的胜率
        total_completion = self.playerA.completion + self.playerB.completion
        playerA_win_rate = self.playerA.completion / total_completion
        playerB_win_rate = self.playerB.completion / total_completion

        return playerA_win_rate, playerB_win_rate

    def plot_win_rate(self):
        # 绘制胜率的折线图
        playerA_win_rate, playerB_win_rate = self.calculate_win_rate()
        plt.plot([playerA_win_rate, playerB_win_rate])
        plt.xlabel('Time')
        plt.ylabel('Win Rate')
        plt.title('Player Win Rate Over Time')
        plt.legend(['Player A', 'Player B'])
        plt.show()


tennisMatchData = TennisMatchData()
file_path = '..\\data\\R7M1.csv'
tennisMatchData.read_data_from_excel(file_path)
tennisMatchData.plot_completion()
tennisMatchData.calculate_win_rate()
tennisMatchData.plot_win_rate()
