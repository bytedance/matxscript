# Generate doc

1. if changed, regenerate：
```bash
cd docs
sphinx-apidoc -f -o source/operators_api/ ../python/byted_vision/ ../python/setup.py ../python/byted_vision/base 
```

2. compile：
```bash
cd docs
make clean
make html
```

3. run local server for debugging：
```bash
cd build/html
python -m http.server
```
