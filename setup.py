import os
import sys

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.5.2'


include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

# compatibility when run in python_bindings
bindings_dir = 'python_bindings'
if bindings_dir in os.path.basename(os.getcwd()):
    source_files = ['./bindings.cpp']
    include_dirs.extend(['../hnswlib/'])
else:
    source_files = ['./python_bindings/bindings.cpp']
    include_dirs.extend(['./hnswlib/'])


libraries = []
extra_objects = []


ext_modules = [
    Extension(
        'hnswlib',
        source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        language='c++',
        extra_objects=extra_objects,
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3', '-march=native'],  # , '-w'
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append("-fopenmp")
        link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))

        build_ext.build_extensions(self)


setup(
    name='hnswlib',
    version=__version__,
    description='hnswlib',
    author='Yury Malkov and others',
    url='https://github.com/yurymalkov/hnsw',
    long_description="""hnsw""",
    ext_modules=ext_modules,
    install_requires=['numpy'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
