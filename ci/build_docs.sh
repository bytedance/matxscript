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

sphinx-build --version


pushd ${ROOT_PATH}/docs

make clean
make html
sphinx-build -D language=zh_CN -b html ./source build/html/zh-CN

popd
