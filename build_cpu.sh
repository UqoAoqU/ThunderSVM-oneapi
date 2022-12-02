rm -rf build

CC=dpcpp
C=icc

mkdir build && cd build && cmake -DCMAKE_C_COMPILER=$C -DCMAKE_CXX_COMPILER=$CC -DCMAKE_EXE_LINKER_FLAGS='-g -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device 0x020a"' -DUSE_GPU=ON -DUSE_DPCPP=ON -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DOpenMP_CXX_FLAGS="-qopenmp -Ofast -g" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libiomp5.so -DCMAKE_MODULE_PATH=. .. && make -j


