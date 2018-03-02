# OpenFST wrapper that supports custom semirings and iPython notebooks

Documentation in `mfst/__init__.py`
Example in `t.py`

Install: `pip install 'git+https://github.com/matthewfl/openfst-wrapper.git'`
This command will take about 5 minutes without printing anything


Develop command: `rm build -rf && CXXFLAGS='-g -DPy_DEBUG' python setup.py install && python t.py`


updating openfst:
    Download the latest version
    mv openfst-1.6.6 openfst
    update the version in setup.py
    run `autoreconf -f -i` in the openfst directory
