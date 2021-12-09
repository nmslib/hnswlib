from pydriller import Repository
import os 
import datetime
os.system("cp examples/speedtest.py examples/speedtest2.py")
for commit in Repository('.', from_tag="v0.5.2").traverse_commits():
    print(commit.hash)
    print(commit.msg)
    
    os.system(f"git checkout {commit.hash}; rm -rf build; ")
    os.system("python -m pip install .")
    os.system(f'python examples/speedtest2.py -n "{commit.msg}" -d 4 -t 1')
    os.system(f'python examples/speedtest2.py -n "{commit.msg}" -d 64 -t 1')
    os.system(f'python examples/speedtest2.py -n "{commit.msg}" -d 128 -t 1')
    os.system(f'python examples/speedtest2.py -n "{commit.msg}" -d 4 -t 24')
    os.system(f'python examples/speedtest2.py -n "{commit.msg}" -d 128 -t 24')

