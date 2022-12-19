Updating English Docs 
================================
1. modify rsts under `docs/source/tutorial_doc`
2. run `ci/build_doc.sh` or `cd docs && make clean && make html` to build english under `build/html`
3. please update Chinese version accordingly.

Updating Chinese Docs
================================
1. run `cd docs && make gettext && sphinx-intl update -p ./build/gettext -l zh_CN` This generates the intermidiate products under `source/locale/zh_CN/LC_MESSAGES`
2. find the parts you modified in the English version.
3. run `sphinx-build -D language=zh_CN -b html ./source build/html/zh-CN` to build the chinese version under `build/html/zh-CN`