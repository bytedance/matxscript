#! /usr/bin/env bash

set -xue
set -o pipefail


THIS_PATH=$(
    cd $(dirname "$0")
    pwd
)

ROOT_PATH=${THIS_PATH}/..


pip3 install sphinx
pip3 install sphinx_rtd_theme
pip3 install sphinx_autorun
pip3 install sphinx-gallery
pip3 install prettytable
pip3 install recommonmark

export PATH=/usr/local/python3.7/bin/:$PATH

sphinx-build --version


pushd ${ROOT_PATH}/docs


sphinx-apidoc -M -e -f --implicit-namespaces -o source/apidoc/ ${ROOT_PATH}/python/matx/ ${ROOT_PATH}/python/setup.py

pushd source/apidoc/

rm matx.vision.tv_transforms.*.rst
rm matx.vision.*.rst

popd 

make clean
make html

popd
