Text2Ids Example
=============================

```shell
# build & dump workflow
python text2ids.py

# build c++ loader
MX_CFLAGS=$(python3 -c 'import matx; print( " ".join(matx.get_cflags()) ) ' )
MX_LINK_FLAGS=$(python3 -c 'import matx; print( " ".join(matx.get_link_flags()) ) ' )
RUNTIME_PATHS=$(python3 -c 'import matx; print( " ".join(["-Wl,-rpath," + p for p in matx.cpp_extension.library_paths()]) )')
g++ -O2 -fPIC -std=c++14 $MX_CFLAGS $MX_LINK_FLAGS ${RUNTIME_PATHS} text2ids.cc -o text2ids

# reload and run by cpp
./text2ids ./my_text2ids
```
