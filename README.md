### **Baseline**

**Complie options**

icpx

***Run***

../build/bin/thundersvm-train -s 0 -t 2 -g 1 -c 10 -o 1 ../data/demo

| main loop |
| --------- |
| 240.036s  |

### **启用多核**

**Complie options**

icpx 

***Run***

../build/bin/thundersvm-train -s 0 -t 2 -g 1 -c 10 -o -1 ../data/demo

| main loop |
| --------- |
| 40.504s   |

dpcpp需要一些额外的操作才可以

这是改过的build_cpu.sh

```sh
rm -rf build

CC=dpcpp

mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=$CC -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_PAPI=OFF -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libiomp5.so -DCMAKE_MODULE_PATH=. .. && make -j



```

### GPU

#### Compile options

-qopenmp -Ofast -fp-model=precise

#### Run

../build/bin/thundersvm-train -s 0 -t 2 -g 1 -c 10 -o -1 ../data/demo

进度：将代码大部分改成了dpcpp，除了sort的部分。

| main loop |
| --------- |
| 26.86s    |

#### 方向

1. 把sort的部分改到GPU上。
2. 代码细节优化下，把内存优化一下
3. 考虑多卡的并行。

#### 用计时函数做了一个profile

| function            | data1        | demo       |
| ------------------- | :----------- | ---------- |
| sort                | 1118.425ms   | 994.797ms  |
| c_smo_solve         | 2841.801ms   | 2579.618ms |
| update_f            | 199.624ms    | 173.002ms  |
| dns_csr_mul         | 166517.711ms | 4326.598ms |
| RBF_kernel          | 251.615ms    | 174.855ms  |
| get_working_set_ins | 1452.001ms   | 323.082ms  |

看起来csmosolve，sort，update_f，优化空间相对较小

但是这个dns_csr_mul又用的mkl，

可以分析一下它的数据流向，然后尝试进行多卡并行。

#### 多卡并行

代码初步完成了，目前的时间是63s，优化了1.5倍（相较于单卡版本）



#### 目前方向：

1.使用vtune进行测试，目前成功的指令是 vtune -collect=gpu-offload ./run-data1.sh，使用之前可能需要加载一下vtune的环境

```
source /opt/intel/oneapi/vtune/2022.1.0/env/vars.sh
```

2.对select_working_set进行微弱的优化，重叠一下（未完成）





### 11.14

经过漫长的调试，多卡目前的时间为59.2s。

主要手段为添加AoT编译，预分配一些内存，减少内存的分配

同时，在尝试切分矩阵的时候发现了一些小问题。

​	1.mkl只在一个compute上跑，然而一个tile貌似可以管4个，这导致GPU总体利用率不高。

2. 测试代码里，在加上openmp之后，GPU上的调度似乎还有点问题，没有最大化利用资源。

2.切分矩阵目前还没有成功，正在查找bug，感觉应该是内存越界的问题（自己造的孽。。。

3.尝试使用gdb-oneapi来帮助寻找bug，最后一次跑成功了，但是中间结果错了，（之前将Re_c独立了出来）

4.考虑下PPT的事情，以及那个自主技术创新申报表的事情。

### 11.16

目前时间52.807s

直接将set_csr_data进行了一个预处理，减少了反复加载数据。

通过观察Vtune的结果以及运行过程中GPU的利用率，认为csr_dns的矩阵乘可能对访存依赖较大，多开线程提升可能也不是很显著。

就没有继续考虑openmp了.
