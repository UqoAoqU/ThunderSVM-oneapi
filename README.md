# PAC22-Thundersvm 

### 概览：

本项目将[thundersvm](https://github.com/Xtra-Computing/thundersvm)进行了移植，使其可以运行在多块intel GPU上。项目重点在于如何使用oneapi套件对cuda代码进行移植优化，以及实现现版本intelGPU的多卡计算。在项目的移植与优化过程中，使用到的开发套件有dpcpp，oneMKL，以及dpcpp library。同时主要使用了intel Vtune进行项目代码的调优。

### 代码移植：

1. 普通kernel函数的移植

   cuda代码的格式与dpcpp的格式基本一致，大多数情况下可以在不修改GPU核心计算代码的情况下完成移植。但需要注意的是目前dpcpp还不支持动态并行，所以部分代码需要进行调整。

   例如：

   ```cuda
   __global__ void kernel() {
   	...
   	int j1 = get_block_min(f_val2reduce, f_idx2reduce);
   	...
   }
   ```

   移植后为

   ```c++
   void kernel() {
       ...
       for (int s = lim / 2; s; s >>= 1) {
           if (tid < s){
               if(f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                    f_idx2reduce[tid] = f_idx2reduce[tid + s];
               } 
       	itm.barrier();
       }
       int j1 = f_idx2reduce[0];
       ...
   }
   ```

2. 数学函数的移植

   oneMKL在大多数数学运算上已支持intel GPU，使用oneMKL在算法的速度与质量上都有保证。本项目中，使用到了oneMKL的稀疏矩阵库用来替代cuSPARSE。注意：cuda代码中矩阵采用的是列优先，dpcpp中矩阵采用的是行优先。

   cuda代码

   ```cpp
   void dns_csr_mul(){        
   	if (!cusparse_init) {
               cusparseCreate(&handle);
               cusparseCreateMatDescr(&descr);
               cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
               cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
               cusparse_init = true;
           }
       kernel_type one(1);
       kernel_type zero(0);
   	cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                           m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                           csr_col_ind.device_data(),
                           dense_mat.device_data(), n, &zero, result.device_data(), m);
       cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                           m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                           csr_col_ind.device_data(),
                           dense_mat.device_data(), n, &zero, result.device_data(), m);
   }
   ```

   dpcpp代码

   ```cpp
   void multidevice_dns_csr_mul() {
   		mkl::sparse::matrix_handle_t handle;
           mkl::sparse::init_matrix_handle(&handle);    
           const kernel_type one(1);
           const kernel_type zero(0);
           mkl::sparse::set_csr_data(handle, m, k, mkl::index_base::zero,
                                       csr_row_ptr, csr_col_ind, csr_val);
           thunder::device_pool[devi].wait();
           auto e = mkl::sparse::gemm(thunder::device_pool[devi], mkl::layout::col_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle, 
                                   dense_mat, n, k, 
                                   zero, result, m);
           mkl::sparse::release_matrix_handle(&handle, {e});
   }
   ```

​	3.基本库函数的移植

​		dpcpp library针对STL库进行了优化，同时也支持intel GPU，这里以sort为例进行演示。

​		cuda代码

```
 void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort) {
     thrust::sort_by_key(thrust::cuda::par, f_val2sort.device_data(), f_val2sort.device_data() + f_val2sort.size(), f_idx2sort.device_data(), thrust::less<float_type>());
 }
```

​		dpcpp代码：

   ```cpp
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <CL/sycl.hpp>
//头文件需置于<algorithm>之前
    void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort) {
        //LOG(INFO) << "sort_f";
     //   sort_t.start();
        buffer<float_type> vals_buf{f_val2sort.device_data(),range<1>(f_val2sort.size())};
        buffer<int> idx_buf{f_idx2sort.device_data(), range<1> (f_idx2sort.size())};
        auto vals_begin = oneapi::dpl::begin(vals_buf);
        auto idx_begin = oneapi::dpl::begin(idx_buf);
        
        oneapi::dpl::execution::__dpl::device_policy<> policy{thunder::device_pool[0]};
        auto zipped_begin = oneapi::dpl::make_zip_iterator(vals_begin, idx_begin);
        std::sort(policy, zipped_begin, zipped_begin + f_val2sort.size(),
                        [](auto lhs, auto rhs) {return get<0>(lhs) < get<0>(rhs); });
     //   sort_t.end();
    }
   ```

​		4.多卡内存管理

​		目前intel GPU并不支持卡间通信，所以必须要通过CPU进行数据的传输。在多卡计算时，最好不要使用shared_ptr，否则会出现错误。

```cpp
   for (int i = 0; i < poolsize; ++i) {
      	device_pool[i].memcpy(val_c[i][0], val_.host_data(), val_.mem_size());
        device_pool[i].memcpy(row_ptr_c[i][0], row_ptr_.host_data(), row_ptr_.mem_size());
        device_pool[i].memcpy(col_ind_c[i][0], col_ind_.host_data(), col_ind_.mem_size());
   }
```

### 性能分析

​		优化过程中，我们使用的主要工具是VTune，这里进行简单演示。

```
vtune -collect gpu-offload -- ./run.sh data
```

​		之后将分析数据用图形化界面打开便可以进行相关的分析。gpu-offload选项可以分析GPU上的热点，GPU各时段的使用状态，以及GPU上不同层级内存的数据传输量。更多使用细节可以参照intel的[官方教程](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/tools/vtune.html)，或是使用help命令来进行查找。

### 运行环境：

*  intel oneapi 2022.1.0

### 编译：

1. 启用oneapi环境

   ```shell
   source /path/to/your/intel/oneapi/setvars.sh
   ```
2. 打开build.sh，配置cmake中intel openmp库路径
   ```shell
   DOpenMP_libiomp5_LIBRARY=/path/to/your/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libiomp5.so
   ```

3. 运行build.sh，完成编译。

### 运行：

将对应格式的数据置于data目录下，然后前往run目录，运行run.sh脚本

```shell
./run.sh ${dataname}
```

运行结果存放于run目录下的result.txt中。
