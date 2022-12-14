#! /usr/bin/env bash
# https://www.sphinx-doc.org/en/master/usage/advanced/intl.html#translating
# https://docs.readthedocs.io/en/stable/guides/manage-translations-sphinx.html
set -xue
set -o pipefail


THIS_PATH=$(
    cd $(dirname "$0")
    pwd
)

ROOT_PATH=${THIS_PATH}/..


pip3 install sphinx-intl

pushd ${ROOT_PATH}/docs
make gettext

sphinx-intl update -p ./build/gettext -l zh_CN

sphinx-build -D language=zh_CN -b html ./source build/html/zh-CN


popd