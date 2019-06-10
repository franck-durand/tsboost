TSBoost, Time Series Boosting
=============================


Context
-------

TSBoost is a framework for time series forecasting.

It mixes classical statistics practices with non linear optimisation techniques of current Machine Learning.

Requirements
------------

32-bit Python is not supported. Please install 64-bit version.


TSBoost uses gradient boosting optimisation provided by `XGBoost <https://github.com/dmlc/xgboost>`_ & `LightGBM <https://github.com/microsoft/LightGBM>`_, both have C++ source code and need a compiler.


For **Windows** users, `VC runtime <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ is needed if **Visual Studio** (2015 or newer) is not installed.


For **Linux** users, **glibc** >= 2.14 is required

    sudo apt-get install build-essential      # Ubuntu/Debian

    sudo yum groupinstall 'Development Tools' # CentOS/RHEL

For **macOS** users, install OpenMP librairy

    brew install libomp

Installation
------------

After installing the compiler, install from `PyPI <https://pypi.org/project/tsboost>`_ Using ``pip``


    pip install tsboost


Quick Start
-----------

You can get started with a jupyter notebook tutorial : `TSBoot quick start <https://github.com/franck-durand/tsboost/jupyter/tsboost_quick_start.ipynb>`_



