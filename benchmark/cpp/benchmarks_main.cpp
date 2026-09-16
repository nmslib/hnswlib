#include <benchmark/benchmark.h>

#include "benchmarks.h"

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    
    RegisterHnswBasicBenchmarks();
    
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    
    return 0;
}