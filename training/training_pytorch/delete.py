import os
import glob


def delete_files_with_pattern(folder_path, pattern):
    # 搜索符合指定模式的文件
    files = glob.glob(os.path.join(folder_path, f"*{pattern}*"))

    # 删除文件
    for file in files:
        os.remove(file)


# 指定文件夹路径和包含的字符串
folder_path = "/date1/ls/training_pytorch/output/simply_64_128/"
pattern = "_image_gaussian_"

# 删除符合条件的文件
delete_files_with_pattern(folder_path, pattern)
