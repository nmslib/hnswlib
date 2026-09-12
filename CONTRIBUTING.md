# Contributing to hnswlib

## Quick start

```bash
gh repo fork nmslib/hnswlib --clone && cd hnswlib
git fetch upstream
git checkout -b my-feature upstream/develop
# ... make changes ...
make install && make test && make examples
git add -u && git commit -m "Short description of change"
git push -u origin HEAD
gh pr create --base develop --fill
```

GitHub's default base is `master`. PRs must target **`develop`**. After opening the PR, leave **Allow edits from maintainers** checked so maintainers can push to your branch.

---

## Setup

[GitHub CLI](https://cli.github.com/) creates the fork, clones it, and configures remotes (`origin` = your fork, `upstream` = nmslib/hnswlib):

```bash
gh repo fork nmslib/hnswlib --clone && cd hnswlib
git fetch upstream
git checkout -b my-feature upstream/develop
```

## Build

**Python** (bindings + C++ extension):
```bash
make install    # pip install .
```

Set `HNSWLIB_NO_NATIVE=1` before `make install` to disable `-march=native` in the Python extension (portable wheels / CI-like builds). This env var does not affect the CMake C++ build.

**C++** (examples and tests only):
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
```

## Test

**Python** — run from the repository root (install the package first):
```bash
make test              # tests/python/bindings_test*.py
make examples          # examples/python/example*.py (same glob as CI)
```

**C++** — run from `build/`:
```bash
ctest --output-on-failure
```

`ctest` does not run `test_updates` (it needs generated data). If you touched updates / replace-deleted:

```bash
# from repository root
python tests/cpp/update_gen_data.py   # needs numpy
# from build/
./test_updates
./test_updates update
```

**Single test** — for fast iteration while debugging:
```bash
# Python: file, class, or method
python -m unittest tests.python.bindings_test_recall
python -m unittest tests.python.bindings_test_recall.RandomSelfTestCase
python -m unittest tests.python.bindings_test_filter.RandomSelfTestCase.testRandomSelf

# C++: by name or regex (from build/)
./searchKnnWithFilter_test
ctest -R searchKnnWithFilter
```

**Exception modes** — CI builds with both; test locally if you touched error handling (from `build/`):
```bash
cmake .. -DHNSWLIB_ENABLE_EXCEPTIONS=ON  && make && ctest --output-on-failure
cmake .. -DHNSWLIB_ENABLE_EXCEPTIONS=OFF && make && ctest --output-on-failure
```

**Sanitizers** (Linux, GCC/Clang; from `build/`):
```bash
cmake .. -DENABLE_ASAN=ON -DENABLE_UBSAN=ON && make && ctest --output-on-failure
```

**Full C++ matrix** — clang/gcc × sanitizers × build types (needs Ninja; from repository root):
```bash
./test_all_build_types.sh
```

## Submit

```bash
git add -u    # plus `git add` any new files (tests, examples)
git commit -m "Short description of change"
git fetch upstream && git rebase upstream/develop
git push -u origin HEAD
gh pr create --base develop --fill
```

Open the PR against **`develop`**, not `master`. `gh pr create --base develop` sets this; on the GitHub website, change the base dropdown.

**Allow maintainer pushes.** When creating the PR, keep **Allow edits from maintainers** checked (GitHub may label it **Allow edits and access to secrets by maintainers** if the fork contains Actions workflows). That lets nmslib/hnswlib maintainers push commits to your PR branch for small fixes, CI, or a rebase. `gh pr create` enables this by default; do not pass `--no-maintainer-can-modify`. Confirm the checkbox on the PR page after creation.

CI runs the matrix automatically (3 OS × Python 3.9–3.13; C++ across OS × compilers × sanitizers × exception modes, plus a C++17 job). All checks must pass before merge. If CI fails — fix locally, push again.

**Before submitting, verify:**

- [ ] Branch is based on `develop` (not `master`)
- [ ] `make test` and `make examples` pass
- [ ] C++ changes: `ctest` passes with both `HNSWLIB_ENABLE_EXCEPTIONS` ON and OFF
- [ ] New functionality includes tests
- [ ] No new external dependencies in `hnswlib/`
- [ ] **Allow edits from maintainers** is checked so maintainers can push to the PR branch

---

## Architecture and conventions

### C++ core — `hnswlib/`

Header-only C++11 with no external dependencies. All changes here automatically benefit every language binding.

| Constraint | Detail |
|------------|--------|
| API pattern | Dual interface: throwing (`addPoint`, `searchKnn`) and non-throwing (`*NoExceptions`) returning `Status`/`StatusOr` |
| SIMD | Behind `USE_SSE`/`USE_AVX`/`USE_AVX512` compile-time guards; the fastest compiled kernel is selected at runtime (`AVXCapable` / `AVX512Capable`) |
| Compiler warnings | Must pass `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang) and `/W1` (MSVC) |
| Threading | Concurrent inserts are safe with inserts (`addPoint` / Python `add_items`); concurrent search is safe with search (`searchKnn` / Python `knn_query`); mixing insert and search is **not** safe; serialization is **not** safe with inserts. New mutable state → use existing locks (`link_list_locks_`, `label_op_locks_`, `global`) |

### Python bindings — `python_bindings/bindings.cpp`

pybind11 wrapper: `Index` → `HierarchicalNSW<float>`, `BFIndex` → `BruteforceSearch<float>`.
Cosine normalization is in the binding layer. No OpenMP on macOS — own `ParallelFor`.

### Where to add tests

| Change | Location | Registration |
|--------|----------|-------------|
| C++ core | `tests/cpp/*_test.cpp` | `CMakeLists.txt` → `TEST_NAMES` |
| C++ updates | `tests/cpp/updates_test.cpp` | special-cased as `test_updates` (not in `TEST_NAMES` / `ctest`) |
| Python bindings | `tests/python/bindings_test_*.py` | automatic via `discover` |
| C++ example | `examples/cpp/*.cpp` | `CMakeLists.txt` → `EXAMPLE_NAMES` |
| Python example | `examples/python/example*.py` | glob in `make examples` and CI |

### CMake options

| Option | Default | Purpose |
|--------|---------|---------|
| `HNSWLIB_ENABLE_EXCEPTIONS` | ON | Compile with/without C++ exceptions |
| `HNSWLIB_CXX_STANDARD` | 11 | C++ dialect (CI also builds 17) |
| `ENABLE_ASAN` | OFF | AddressSanitizer |
| `ENABLE_UBSAN` | OFF | UndefinedBehaviorSanitizer |
| `ENABLE_TSAN` | OFF | ThreadSanitizer (not in the default CI matrix) |

---

## Language bindings and integrations

hnswlib's header-only design makes FFI straightforward. Existing community bindings are listed under [Other implementations](README.md#other-implementations) in README.md.

**Creating a new binding** (Zig, Ruby, Scala, Swift, etc.): prefer linking headers directly or via a thin C shim (`extern "C"`). Use `saveIndex`/`loadIndex` or the stream-based interface for cross-language index portability. Test at minimum: construction, recall, serialization round-trip, filtered search. We are happy to list your project in README.md.

## Reporting issues

Search [existing issues](https://github.com/nmslib/hnswlib/issues) first. Include: hnswlib version, OS, compiler/Python version, reproducing code. For recall problems — your `M`, `ef_construction`, `ef`, `dim`, and dataset size.
