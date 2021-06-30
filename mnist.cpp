/*
  A simple example, tested under Linux.
  first, copy train-labels.idx1-ubyte, train-images.idx3-ubyte, t10k-labels.idx1-ubyte, t10k-images.idx3-ubyte to the same path of the executable file.
  then, ./mnist
*/
#include <fstream>
#include <queue>
#include <chrono>
#include <unistd.h>
#include <sys/resource.h>
#include "hnswlib/hnswlib.h"

using namespace hnswlib;

//Returns the current resident set size (physical memory use) measured in Mb.
static size_t getCurrentRSS(){
    FILE *fp = fopen("/proc/self/statm", "r");
    if(fp == NULL) return (size_t)0L;
    long rss;
    if(fscanf(fp, "%*s%ld", &rss) != 1){
        fclose(fp);
        return (size_t)0L;
    }
    fclose(fp);
    return (size_t)rss*sysconf(_SC_PAGESIZE)/1024/1024;
}

inline bool open_check(const char *name){
	FILE *fp = fopen(name, "r");
	if(fp == NULL) return false;
    fclose(fp);
    return true;
}

int main(int argc, char *argv[]){
    if(!open_check("train-labels.idx1-ubyte") || !open_check("train-images.idx3-ubyte") ||
       !open_check("t10k-labels.idx1-ubyte") || !open_check("t10k-images.idx3-ubyte")){
		printf("open mnist files error.\n");
		return 0;
    }
	unsigned int efConstruction = 48, M = 16, vecdim = 784;
    printf("efConstruction=%u, M=%u, vecdim=%u\n", efConstruction, M, vecdim);

	FILE *fp = fopen("train-labels.idx1-ubyte", "rb");
	unsigned char ubyte[16];
	size_t sz = fread(ubyte, 1, 8, fp); //ubyte[0-3] = 0X0081
	unsigned int trainSize = 16777216*ubyte[4] + 65536*ubyte[5] + 256*ubyte[6] + ubyte[7];
	unsigned char *trainCls = new unsigned char[trainSize];
	sz = fread(trainCls, 1, trainSize, fp);
	fclose(fp);

    fp = fopen("train-images.idx3-ubyte", "rb");
    sz = fread(ubyte, 1, 16, fp);
	unsigned char *trainSet = new unsigned char[trainSize*vecdim];
	for(unsigned int i=0; i<trainSize; i++){
        sz = fread(&trainSet[i*vecdim], 1, vecdim, fp);
    }
    fclose(fp);

    L2SpaceI l2space(vecdim);
    HierarchicalNSW<int> *appr_alg;
    std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
    if(open_check("mnist.bin")){
        printf("Loading index from mnist.bin:\n");
        appr_alg = new HierarchicalNSW<int>(&l2space, "mnist.bin");
        printf("Actual memory usage: %d Mb\n", getCurrentRSS());
    }else{
        printf("Building index:\n");
        appr_alg = new HierarchicalNSW<int>(&l2space, trainSize, M, efConstruction);
        #pragma omp parallel for
        for(unsigned int i = 0; i < trainSize; i++){
            appr_alg->addPoint((void*)&trainSet[i*vecdim], (size_t)trainCls[i]);
        }
        printf("Build time: %fs\n", 0.000001*std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_begin).count());
        appr_alg->saveIndex("mnist.bin");
    }

    printf("testing...\n");
	fp = fopen("t10k-labels.idx1-ubyte", "rb");
	sz = fread(ubyte, 1, 8, fp); //ubyte[0-3] = 0X0081
	unsigned int testSize = 16777216*ubyte[4] + 65536*ubyte[5] + 256*ubyte[6] + ubyte[7];
	unsigned char *testCls = new unsigned char[testSize];
	sz = fread(testCls, 1, testSize, fp);
	fclose(fp);

	fp = fopen("t10k-images.idx3-ubyte", "rb");
	sz = fread(ubyte, 1, 16, fp);
	unsigned char *testSet = new unsigned char[testSize*vecdim];
	for(unsigned int i=0; i<testSize; i++){
        sz = fread(&testSet[i*vecdim], 1, vecdim, fp);
    }
	fclose(fp);

    unsigned int k = 1;
    for(int ef = 8; ef < 32; ef++){
        appr_alg->setEf(ef);
        time_begin = std::chrono::steady_clock::now();
        unsigned int correct = 0, total = 0;
        #pragma omp parallel for reduction(+:total,correct)
        for(unsigned int i = 0; i < testSize; i++){
            total += k;
            std::priority_queue<std::pair<int, labeltype>> result = appr_alg->searchKnn(testSet + vecdim * i, k);
            while(result.size()){
                if(result.top().second == testCls[i]) correct++;
                result.pop();
            }
        }
        printf("ef=%d, recall=%f, time=%.3fms\n", ef, double(correct)/total, 0.001*std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - time_begin).count());
    }
    printf("Actual memory usage: %dMb\n", getCurrentRSS());

	delete[] trainCls;
	delete[] trainSet;
	delete[] testCls;
	delete[] testSet;
    return 1;
}
