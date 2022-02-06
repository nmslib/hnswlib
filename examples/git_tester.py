from pydriller import Repository
import os 
import datetime
os.system("cp examples/speedtest.py examples/speedtest2.py") # the file has to be outside of git
for idx, commit in enumerate(Repository('.', from_tag="v0.6.0").traverse_commits()):    
    name=commit.msg.replace('\n', ' ').replace('\r', ' ')
    print(idx, commit.hash, name)



for commit in Repository('.', from_tag="v0.6.0").traverse_commits():
    
    name=commit.msg.replace('\n', ' ').replace('\r', ' ')
    print(commit.hash, name)
    
    os.system(f"git checkout {commit.hash}; rm -rf build; ")
    print("\n\n--------------------\n\n")
    ret=os.system("python -m pip install .")
    print(ret)
    
    if ret != 0:
        print ("build failed!!!!")
        print ("build failed!!!!")
        print ("build failed!!!!")
        print ("build failed!!!!")
        continue    
    
    os.system(f'python examples/speedtest2.py -n "{name}" -d 4 -t 1')
    os.system(f'python examples/speedtest2.py -n "{name}" -d 64 -t 1')
    os.system(f'python examples/speedtest2.py -n "{name}" -d 128 -t 1')
    os.system(f'python examples/speedtest2.py -n "{name}" -d 4 -t 24')
    os.system(f'python examples/speedtest2.py -n "{name}" -d 128 -t 24')


