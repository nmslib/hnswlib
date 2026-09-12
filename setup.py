import os
import sys
import platform

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.10.0'


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


def _normalize_cxx_std(value):
    """Return 11, 14, 17, 20, or 23 from env/user input like '17' or 'c++17'."""
    text = str(value).strip().lower()
    if text.startswith('gnu++'):
        text = text[5:]
    elif text.startswith('c++'):
        text = text[3:]
    allowed = ('11', '14', '17', '20', '23')
    if text not in allowed:
        raise RuntimeError(
            'HNSWLIB_CXX_STD must be one of %s (got %r)'
            % (', '.join(allowed), value))
    return int(text)


def requested_cxx_std():
    """C++ standard requested for the Python extension. Default is 11."""
    return _normalize_cxx_std(os.environ.get('HNSWLIB_CXX_STD', '11'))


def _std_compile_flag(compiler, std):
    """Compiler flag for ISO C++ `std`, or None if the compiler needs no flag."""
    if compiler.compiler_type == 'msvc':
        # MSVC has no /std:c++11; VS 2015+ is C++11 without an extra flag.
        if std == 11:
            return None
        return '/std:c++%d' % std
    return '-std=c++%d' % std


def cpp_flag(compiler):
    """Probe the flag for the requested C++ standard.

    Default is C++11. Set HNSWLIB_CXX_STD=14|17|20|23 to request a newer
    dialect. Falls back only downward if the compiler rejects the request.
    Does not silently prefer C++14 when C++11 was requested.
    """
    requested = requested_cxx_std()
    chosen_std = None
    chosen_flag = None
    for std in (23, 20, 17, 14, 11):
        if std > requested:
            continue
        flag = _std_compile_flag(compiler, std)
        if flag is None or has_flag(compiler, flag):
            chosen_std = std
            chosen_flag = flag
            break
    if chosen_std is None:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')
    if chosen_std != requested:
        print('Requested C++%d is not supported; falling back to C++%d'
              % (requested, chosen_std))
    if chosen_flag is None:
        print('hnswlib C++ standard: C++%d (compiler default, no extra flag)'
              % chosen_std)
    else:
        print('hnswlib C++ standard: C++%d (%s)' % (chosen_std, chosen_flag))
    return chosen_flag


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
            std_flag = cpp_flag(self.compiler)
            if std_flag:
                opts.append(std_flag)
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
            # Enable exceptions.
            opts.append("-fexceptions")
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            # Enable exceptions.
            opts.append('/EHsc')
            std_flag = cpp_flag(self.compiler)
            if std_flag:
                opts.append(std_flag)
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
