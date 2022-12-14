.. Installating From Source

Installating From Source
#############################################

Prerequisite
************************************
| CMAKE >=3.2   
| gcc>=6.0

Build Matx using script
************************************
.. code-block:: bash 

    # step 1: obtain the source code
    git clone https://github.com/bytedance/matxscript
    # step 2: build using script
    cd matxscript && bash ci/build_pip_whl.sh
    # step 3: install Python package
    cd output && pip install .

Verify installation
************************************
.. code-block:: python3 

    import matx
    print(matx.__version__)

