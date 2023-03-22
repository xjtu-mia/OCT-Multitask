import os
import shutil
def copytree(src, dst):
    """
    递归地复制文件夹src及其子文件夹中的所有文件到dst目录。
    如果dst目录中存在同名文件夹，则将src中的文件夹合并到dst中的文件夹中。
    如果dst目录中存在同名文件，则用src中的文件替换dst中的文件。
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        src_file = os.path.join(src, item)
        dst_file = os.path.join(dst, item)
        if os.path.isdir(src_file):
            # 处理同名文件夹的情况
            if os.path.exists(dst_file):
                copytree(src_file, dst_file)
            else:
                shutil.copytree(src_file, dst_file)
        else:
            # 处理同名文件的情况
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy2(src_file, dst_file)