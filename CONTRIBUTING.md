# Contributing to hnswlib

```bash
gh repo fork nmslib/hnswlib --clone && cd hnswlib
git checkout -b my-feature upstream/develop
# ... make changes ...
make install && make test && make examples
git add -u && git commit -m "Short description of change"
gh pr create --base develop --fill
```

---

## Setup

[GitHub CLI](https://cli.github.com/) creates the fork, clones it, and configures remotes (`origin` = your fork, `upstream` = nmslib/hnswlib):

```bash
gh repo fork nmslib/hnswlib --clone && cd hnswlib
git checkout -b my-feature upstream/develop
```

## Build

**Python** (bindings + C++ extension):
```bash
make install
```

**C++** (examples and tests only):
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
```

Set `HNSWLIB_NO_NATIVE=1` before `make install` to disable `-march=native` for portable builds.

## Test

**Python** — run from the repository root:
```bash
make test              # all tests
make examples          # all examples from examples/python/
```

**C++** — run from `build/`:
```bash
ctest --output-on-failure
```

**Single test** — for fast iteration while debugging:
```bash
# Python: file, class, or method
python -m unittest tests.python.bindings_test_recall
python -m unittest tests.python.bindings_test_recall.RandomSelfTestCase
python -m unittest tests.python.bindings_test_filter.RandomSelfTestCase.testRandomSelf

# C++: by name or regex
./searchKnnWithFilter_test
ctest -R searchKnnWithFilter
```

**Exception modes** — CI builds with both; test locally if you touched error handling:
```bash
cd build
cmake .. -DHNSWLIB_ENABLE_EXCEPTIONS=ON  && make && ctest --output-on-failure
cmake .. -DHNSWLIB_ENABLE_EXCEPTIONS=OFF && make && ctest --output-on-failure
```

**Sanitizers** (Linux, GCC/Clang):
```bash
cmake .. -DENABLE_ASAN=ON -DENABLE_UBSAN=ON && make && ctest --output-on-failure
```

**Full C++ matrix** — all compilers × sanitizers × build types:
```bash
./test_all_build_types.sh
```

## Submit

```bash
git add -u && git commit -m "Short description of change"
git fetch upstream && git rebase upstream/develop
gh pr create --base develop --fill
```

CI runs the full matrix automatically (3 OS × 5 Python versions; 3 OS × compilers × sanitizers × exception modes). All checks must pass before merge. If CI fails — fix locally, push again.

**Before submitting, verify:**

- [ ] Branch is based on `develop`
- [ ] `make test` and `make examples` pass
- [ ] C++ changes: `ctest` passes with both `HNSWLIB_ENABLE_EXCEPTIONS` ON and OFF
- [ ] New functionality includes tests
- [ ] No new external dependencies in `hnswlib/`

---

## Architecture and conventions

### C++ core — `hnswlib/`

Header-only C++11 with no external dependencies. All changes here automatically benefit every language binding.

| Constraint | Detail |
|------------|--------|
| API pattern | Dual interface: throwing (`addPoint`, `searchKnn`) and non-throwing (`*NoExceptions`) returning `Status`/`StatusOr` |
| SIMD | Behind `USE_SSE`/`USE_AVX`/`USE_AVX512` guards; selected at runtime |
| Compiler warnings | Must pass `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang) and `/W1` (MSVC) |
| Threading | `add_items` safe with `add_items`; `knn_query` safe with `knn_query`; mixing is NOT safe; serialization is NOT safe with `add_items`. New mutable state → use existing locks (`link_list_locks_`, `label_op_locks_`, `global`) |

### Python bindings — `python_bindings/bindings.cpp`

pybind11 wrapper: `Index` → `HierarchicalNSW<float>`, `BFIndex` → `BruteforceSearch<float>`.
Cosine normalization is in the binding layer. No OpenMP on macOS — own `ParallelFor`.

### Where to add tests

| Change | Location | Registration |
|--------|----------|-------------|
| C++ core | `tests/cpp/*_test.cpp` | `CMakeLists.txt` → `TEST_NAMES` |
| Python bindings | `tests/python/bindings_test_*.py` | automatic via `discover` |
| C++ example | `examples/cpp/*.cpp` | `CMakeLists.txt` → `EXAMPLE_NAMES` |

### CMake options

| Option | Default | Purpose |
|--------|---------|---------|
| `HNSWLIB_ENABLE_EXCEPTIONS` | ON | Compile with/without C++ exceptions |
| `ENABLE_ASAN` | OFF | AddressSanitizer |
| `ENABLE_UBSAN` | OFF | UndefinedBehaviorSanitizer |
| `ENABLE_TSAN` | OFF | ThreadSanitizer |

---

## Language bindings and integrations

hnswlib's header-only design makes FFI straightforward. Existing community bindings:

| Language | Projects |
|----------|----------|
| Rust | [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs), [hnsw](https://github.com/rust-cv/hnsw) |
| Java | [hnswlib](https://github.com/jelmerk/hnswlib), [hnswlib-jna](https://github.com/stepstone-tech/hnswlib-jna) |
| Go | [go-hnsw](https://github.com/Bithack/go-hnsw) |
| R | [rcpphnsw](https://github.com/jlmelville/rcpphnsw) |
| .NET | [hnsw-sharp](https://github.com/curiosity-ai/hnsw-sharp) |
| Julia | [HNSW.jl](https://github.com/JuliaNeighbors/HNSW.jl) |

**Creating a new binding** (Zig, Ruby, Scala, Swift, etc.): prefer linking headers directly or via a thin C shim (`extern "C"`). Use `saveIndex`/`loadIndex` or the stream-based interface for cross-language index portability. Test at minimum: construction, recall, serialization round-trip, filtered search. We are happy to list your project in README.md.

## Reporting issues

Search [existing issues](https://github.com/nmslib/hnswlib/issues) first. Include: hnswlib version, OS, compiler/Python version, reproducing code. For recall problems — your `M`, `ef_construction`, `ef`, `dim`, and dataset size.
