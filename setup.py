# from distutils.core import setup, Extension
# from distutils.command import build_ext
from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension
import setuptools
import sys
import os

try:
    import pybind11
except ImportError:
    r = os.system('pip install pybind11')
    assert r
    import pybind11

if (not os.path.exists(os.path.join(sys.prefix, 'include', 'fst'))
    or 'mfl_hack' not in open(os.path.join(sys.prefix, 'include', 'fst', 'script', 'fst-class.h'), 'r').read()):
    # then there are no open fst headers where we expect them
    # -__-
    cmd = "cd {dir} && ./configure CXX='g++ -g' CC='gcc -g' prefix={prefix} --enable-grm --enable-bin --enable-ngram-fsts --enable-compact-fsts && make -j {threads} && make install".format(
        dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openfst'),
        prefix=sys.prefix,
        threads=max(os.cpu_count() - 2, 1)
    )
    print(cmd)
    r = os.system(cmd)
    assert r


ext_modules = [
    Extension(
        'openfst_wrapper_backend',
        ['mfst/openfst_wrapper.cc'],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            pybind11.get_include(user=True),
            # get_pybind_include(),
            # get_pybind_include(user=True)
            os.path.join(sys.prefix, 'include')
        ],
        library_dirs=[os.path.join(sys.prefix, 'lib')],
        runtime_library_dirs=[os.path.join(sys.prefix, 'lib')],
        libraries=['fst', 'fstscript'],  # these paths should be defined to use the path that this library is installed in
        language='c++'
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
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

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
            ext.extra_compile_args = opts
            build_ext.build_extensions(self)

setup(
    name='mfl-openfst',
    version='0.0.1',
    description='Wrap OpenFST with support for defining a custom weight semiring',
    author='Matthew Francis-Landau',
    author_email='matthew@matthewfl.com',
    packages=['mfst'],
    install_requires=['pybind11>=2.2'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
