# Installing Scipy BLAS LAPAC on Mac

Scipy and numpy require installation of BLAS/LAPAC or fail

**scipy**

```
poetry add scipy # finally did the trick since it took scipy>-13.1  
```

**Stackovervlow**

- https://stackoverflow.com/questions/69954587/no-blas-lapack-libraries-found-when-installing-scipy

- Install lapack and openblas from Homebrew

```
$ brew install atlas openblas lapack

```

**Tell Numpy installer where to find lapack**

```
export LDFLAGS="-L/usr/local/opt/lapack/lib"
export CPPFLAGS="-I/usr/local/opt/lapack/include"
export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig"
```

See ... Python 3.9 ... - https://github.com/scipy/scipy/issues/12935
export

```
CFLAGS=-Wno-error=implicit-function-declaration
```

The location may vary - use find command to find this on your local /usr/local/opt

```
export LAPACK=/usr/local/opt/lapack/lib/liblapack.dylib
export BLAS=/usr/local/opt/openblas/lib/libopenblasp-r0.3.19.dylib
```

# mkl - outside of virtual env

```
pip install mkl

---

mkl-2023.2.2

Successfully installed intel-openmp-2023.2.0 mkl-2023.2.2 tbb-2021.10.0
```

# install scipy

python -m pip install scipy==1.6.1