import csv

def compare_csv_columns(file1, file2):
    with open(file1, mode='r', encoding='utf-8') as f1, open(file2, mode='r', encoding='utf-8') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        # 遍历两个文件的行
        for row_num, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            # 对比第一列
            if row1[0] != row2[0]:
                print(f"第 {row_num} 行不一致：")
                print(f"{file1}: {row1}")
                print(f"{file2}: {row2}")
                return  # 找到第一个不一致的地方后退出

        print("所有行的第一列内容一致。")

# 使用示例
file1 = 'answers_organize.csv'
file2 = 'q.csv'
compare_csv_columns(file1, file2)