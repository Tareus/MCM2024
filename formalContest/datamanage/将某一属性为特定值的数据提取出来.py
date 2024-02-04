import pandas as pd


def extract_rows_by_match_id(input_file, output_file, match_id_value):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 提取属性"match_id"的值为指定值的行
    extracted_data = df[df['match_id'] == match_id_value]

    # 保存提取的数据到新的CSV文件
    extracted_data.to_csv(output_file, index=False)


if __name__ == "__main__":
    # 指定输入文件路径
    input_file_path = "..\\data\\data_1.csv"

    # 指定输出文件路径
    output_file_path = "..\\data\\R7M1.csv"

    # 指定匹配的match_id值
    target_match_id = "2023-wimbledon-1701"

    # 调用函数进行提取
    extract_rows_by_match_id(input_file_path, output_file_path, target_match_id)
