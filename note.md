## 背景
hnswlib是用于计算knn(k-nearest neighbors)问题的一个近似算法。想从大批量数据中寻找相似的k个数据，是一个常见的场景。使用范围很广泛。

判断数据相似的一般方式是，先把每条数据提取feature处理成一个vector，每个数据的vector长度相同。然后通过计算距离（可以自定义，比如余弦，欧式距离等）判断两个数据的相似度。

从大量数据中，查找最近k个相似数据的复杂度很高，穷举需要全部遍历所有的数据才行。当然，先聚类再查会剪枝，但并不会降低太多复杂度。

## 算法
hnswlib的核心思想是：
- 把数据分层，每个数据`q`插入时随机生成它所在的层级`l`，该数据和`level <= l`的层级上的数据建立 `M (=16)` 个连接。
- 查询时，从最上层(`levelMax`)给定点`ep`开始，查询`ep`的neighbors，选择最近的点，作为新的`ep`，进入下一层。直到第0层。
- 通过`ep`，在第0层中查询`ep`的neighbors，并选择最近的k条数据返回。

hnswlib的核心代码代码包含4个函数：
- `add_point`: 用于插入一条新的数据
- `searchLayer`: 给定查询点`q`和入口点`ep`，查出最近的m个点
- `getNeighbors`: 给定查询点`q`和一批候选点，查出最近的m个点
- `searchKnn`: 给定查询点`q`，查出最近的k个点。

每个函数的具体细节就不写了。

## 代码存储
存储结构基本可以看做3个部分：
- headers: 包含该图的一些配置参数，比如最多多少个点，最多多少条边，目前有多少个点等
- 第0层的数据: 主要是每个点的基本数据data详情和label名，和在第0层每个点的neighbors
- 其他层的数据: 主要是每个点在其他层的neighbors

第0层的存储结构(`data_level0_memory_`)：
该层先按点切分，每个点一个固定长度`size_data_per_element_`，这个长度中按顺序存了4个内容：
- `neighbor个数`: 固定int长度
- `neighbors`: 最多M个neighbors，所以固定 M 个int长度
- `data`: 数据大小固定长度
- `label`: 固定长度

所以整体看起来就是这样子
[`#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`], `data`, `label1`]
[`#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`], `data`, `label2`]
[`#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`], `data`, `label3`]
...
[`#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`], `data`, `labeln`]

其他层存储（`link_list`）:
也是按点来切分，每个点非固定大小。每个点内部包含两个部分：
- 该点数据的总长度: 固定长度int，(`linkListSize`)
- 该点每一层的neighbors：非固定长度，总长度为`该点层级*每层固定长度(size_links_per_element_)`
每个点每一层的存储结构按顺序为：
- `neighbor个数`: 固定长度
- `neighbors`: 最多M个neighbors，所以固定 M 个int长度

所以整体看起来就是这样子
[`linkListSize`, [`l1#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l2#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l3#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], ...]
[`linkListSize`, [`l1#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l2#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l3#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], ...]
[`linkListSize`, [`l1#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l2#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l3#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], ...]
...
[`linkListSize`, [`l1#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l2#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], [`l3#neighbors`, [`neighbor1`,`neighbor2`,`neighbor3`]], ...]

## 构建编译调试环境
不想在本机搞，不想在开发机上搞，原因是害怕安装软件把把环境搞混乱了。
所以选择在Docker中搞。

1. Docker早就安装好了。
2. 下载安装最新VS
3. 安装C/C++插件
4. 安装Docker插件
5. 准备一个之后长期稳定用的docker镜像
6. 拉起容器

```shell
# C++ 代码全放在/Users/pengchao/git/gcc/ 下
# 数据类放在/Users/pengchao/data 下
docker run -d --name pcworker --rm -v /Users/pengchao/git/gcc/:/gcc -v /Users/pengchao/data:/data pcworker sleep 10000000
```

7. 通过VS Code连接container
通过F1，搜索 “Remote-Container: Attach to Running Container”，最下面可以看到连接状态。

打开想编辑的项目。

找到C/C++ 插件，安装到容器，装完后点击 Reload Container。

8. 添加测试配置
在.vscode下创建两个文件

launch.json
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch Main",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/main.out",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "Build Main"
        }
    ]
}
```

tasks.json
```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build Main",
            "type": "shell",
            "command": "g++ -g main.cpp -o main.out",
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

即可测试。

## 测试
实现了一个Did you mean的功能，自己实现了一个`LevenshteinDistance`的距离算法，并写入了330k个单词，单线程花了10min，并测试了一下执行结果，符合预期。

## 多线程
hnswlib是通过加锁来实现的实现并发写的，也就是对每层每个点的neighbors写加锁。

## 分布式
还每来得及思考。
