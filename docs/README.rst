# 生成文档

1. 如果文档有变化，需要重新生成注释文档：
```bash
cd docs
sphinx-apidoc -f -o source/operators_api/ ../python/byted_vision/ ../python/setup.py ../python/byted_vision/base 
```

2. 编译：
```bash
cd docs
make clean
make html
```

3. 本地调试启动服务器：
```bash
cd build/html
python -m http.server
```