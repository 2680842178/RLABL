import csv

def merge_rows(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)  # 读取表头
        writer.writerow(header)  # 写入表头

        rows = list(reader)
        for i in range(0, len(rows), 2):
            if i + 1 < len(rows):
                merged_row = rows[i] + rows[i + 1]
                writer.writerow(merged_row)
            else:
                writer.writerow(rows[i])

# 使用示例
input_file = './storage/20241111-seed1/log.csv'
output_file = 'merged_log.csv'
merge_rows(input_file, output_file)