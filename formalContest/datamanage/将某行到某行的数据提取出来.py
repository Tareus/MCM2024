import pandas as pd


def extract_rows(input_file, output_file, start_row, end_row):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 提取指定行范围的对象
    extracted_data = df.iloc[start_row - 1:end_row]

    # 保存提取的数据到新的CSV文件
    extracted_data.to_csv(output_file, index=False)


if __name__ == "__main__":
    # 指定输入文件路径
    input_file_path = "..\\data\\data_1.csv"

    # 指定输出文件路径
    output_file_path = "..\\data\\R3M1.csv"

    # 指定提取的起始行和结束行
    start_row_index = 301
    end_row_index = 301

    # 调用函数进行提取
    extract_rows(input_file_path, output_file_path, start_row_index, end_row_index)
