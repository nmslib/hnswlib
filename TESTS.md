
# Tests


### 200M SIFT test reproduction 
To download and extract the bigann dataset:
```bash
python3 download_bigann.py
```
To compile:
```bash
cmake .
make all
```

To run the test on 200M SIFT subset:
```bash
./main
```

The size of the bigann subset (in millions) is controlled by the variable **subset_size_milllions** hardcoded in **sift_1b.cpp**.

### Feature Vector Updates test
To generate testing data (from root directory):
```bash
cd examples
python update_gen_data.py
```
To compile (from root directory):
```bash
mkdir build
cd build
cmake ..
make 
```
To run test **without** updates (from `build` directory)
```bash
./test_updates
```

To run test **with** updates (from `build` directory)
```bash
./test_updates update
```