import os
import sys
import platform

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.8.0'


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
    compiler_flag_native = '-march=native'
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3', compiler_flag_native],  # , '-w'
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if os.environ.get("HNSWLIB_NO_NATIVE"):
        c_opts['unix'].remove(compiler_flag_native)

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append("-fopenmp")
        link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = BuildExt.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if not os.environ.get("HNSWLIB_NO_NATIVE"):
                # check that native flag is available
                print('checking avalability of flag:', BuildExt.compiler_flag_native)
                if not has_flag(self.compiler, BuildExt.compiler_flag_native):
                    print('removing unsupported compiler flag:', BuildExt.compiler_flag_native)
                    opts.remove(BuildExt.compiler_flag_native)
                    # for macos add apple-m1 flag if it's available
                    if sys.platform == 'darwin':
                        m1_flag = '-mcpu=apple-m1'
                        print('checking avalability of flag:', m1_flag)
                        if has_flag(self.compiler, m1_flag):
                            print('adding flag:', m1_flag)
                            opts.append(m1_flag)
                        else:
                            print(f'flag: {m1_flag} is not available')
                else:
                    print(f'flag: {BuildExt.compiler_flag_native} is available')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(BuildExt.link_opts.get(ct, []))

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
