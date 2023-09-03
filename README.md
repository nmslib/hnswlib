# Chroma-Hnswlib - fast approximate nearest neighbor search
Chromas fork of https://github.com/nmslib/hnswlib

## Build & Release

Wheels are automatically built and pushed to PyPI for multiple
platforms via GitHub actions using the
[cibuildwheel](https://github.com/pypa/cibuildwheel).

### Building AVX Extensions

For maximum compatibility, the distributed wheels are not compiled to
make use of Advanced Vector Extensions (AVX). If your hardware
supports AVX, you may get better performance by recompiling this
library on the machine on which it is intended to run.

To force recompilation when installing, specify the `--no-binary
chroma-hsnwlib` option to PIP when installing dependencies. This can
be added to your `pip install` command, for example:

```
pip install -r requirements.txt --no-binary chroma-hnswlib
```

You can also put the `--no-binary` directive [in your requirements.txt](https://pip.pypa.io/en/stable/cli/pip_install/#install-no-binary).

If you've already installed dependencies, you must first uninstall
`chroma-hsnwlib` using `pip uninstall chroma-hnswlib` to remove the
precompiled version before reinstalling.
