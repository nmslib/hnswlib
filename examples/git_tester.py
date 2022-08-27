import os

from sys import platform
from pydriller import Repository

if platform == "win32":
    copy_cmd = "copy"
    rm_dir_cmd = "rmdir /s /q"
else:
    copy_cmd = "cp"
    rm_dir_cmd = "rm -rf"

speedtest_src_path = os.path.join("examples", "speedtest.py")
speedtest_copy_path = os.path.join("examples", "speedtest2.py")
os.system(f"{copy_cmd} {speedtest_src_path} {speedtest_copy_path}")  # the file has to be outside of git

commits = list(Repository('.', from_tag="v0.6.0").traverse_commits())
print("Found commits:")
for idx, commit in enumerate(commits):
    name = commit.msg.replace('\n', ' ').replace('\r', ' ')
    print(idx, commit.hash, name)

for commit in commits:
    name = commit.msg.replace('\n', ' ').replace('\r', ' ')
    print("\nProcessing", commit.hash, name)

    os.system(f"git checkout {commit.hash}")
    os.system(f"{rm_dir_cmd} build")
    print("\n\n--------------------\n\n")
    ret = os.system("python -m pip install .")
    print("Install result:", ret)

    if ret != 0:
        print("build failed!!!!")
        print("build failed!!!!")
        print("build failed!!!!")
        print("build failed!!!!")
        continue

    os.system(f'python {speedtest_copy_path} -n "{name}" -d 4 -t 1')
    os.system(f'python {speedtest_copy_path} -n "{name}" -d 64 -t 1')
    os.system(f'python {speedtest_copy_path} -n "{name}" -d 128 -t 1')
    os.system(f'python {speedtest_copy_path} -n "{name}" -d 4 -t 24')
    os.system(f'python {speedtest_copy_path} -n "{name}" -d 128 -t 24')
