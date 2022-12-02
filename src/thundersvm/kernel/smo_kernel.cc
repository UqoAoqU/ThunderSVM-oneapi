// smo_kernel with dpcpp
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <CL/sycl.hpp>

#include "thundersvm/kernel/smo_kernel.h"
#include "thundersvm/timer.h"
using namespace sycl;
using namespace std;
namespace svm_kernel {
    const size_t Smem_size = 1024 * (sizeof(int) + sizeof(float_type) + sizeof(kernel_type)) + 2 * sizeof(float_type);
    timer c_smo("c_smo_solve");
    timer upf("update_f");
    timer sort_t("sort");
    
    // merged with get_block_min
    void 
    c_smo_solve_kernel(const int *label, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                       const int *working_set, int ws_size,
                       float_type Cp, float_type Cn, const kernel_type *k_mat_rows, const kernel_type *k_mat_diag, int row_len,
                       float_type eps,
                       float_type *diff, int max_iter, size_t smem_size) {    
                        
        float_type *op = malloc_shared<float_type>(10, thunder::device_pool[0]);
        static size_t Glob = ws_size;
        static size_t Loca = ws_size; 
        int*shared_mem = (int*) malloc_device(Smem_size, thunder::device_pool[0]);  
        thunder::device_pool[0].submit([&](sycl::handler &h){
            range<1> glob {Glob};
            range<1> loca {Loca};
            h.parallel_for(nd_range(glob, loca), [=](nd_item<1> itm){ 
                              
                int *f_idx2reduce = shared_mem; //temporary memory for reduction
                float_type *f_val2reduce = (float_type *) &shared_mem[ws_size]; //f values used for reduction.
                float_type *alpha_i_diff = (float_type *) &shared_mem[ws_size + ws_size * sizeof(float_type) / sizeof(int)]; //delta alpha_i
                float_type *alpha_j_diff = &alpha_i_diff[1];
                kernel_type *kd = (kernel_type *) &alpha_j_diff[1]; // diagonal elements for kernel matrix
                int tid = itm.get_local_id(0);
                int wsi = working_set[tid];
                kd[tid] = k_mat_diag[wsi];
                float_type y = label[wsi];
                float_type f = f_val[wsi];
                float_type a = alpha[wsi];
                float_type aold = a;
                itm.barrier();
                float_type local_eps;
                int numOfIter = 0;
                int lim = itm.get_global_range(0);
                while (1) {
                //select fUp and fLow
                    // if (is_I_up(a, y, Cp, Cn))
                    //     f_val2reduce[tid] = f;
                    // else
                    //     f_val2reduce[tid] = INFINITY;
                    f_val2reduce[tid] = (is_I_up(a, y, Cp, Cn)) ? f : INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = lim / 2; s; s >>= 1) {
                        if (tid < s){
                            if(f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                                f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        }
                        itm.barrier();
                    }
                    int i = f_idx2reduce[0];
                    float_type up_value = f_val2reduce[i]; 

                    kernel_type kIwsI = k_mat_rows[row_len * i + wsi];//K[i, wsi]
                    itm.barrier();

                    // if (is_I_low(a, y, Cp, Cn))
                    //     f_val2reduce[tid] = -f;
                    // else
                    //     f_val2reduce[tid] = INFINITY;
                    f_val2reduce[tid] = (is_I_low(a, y, Cp, Cn)) ? -f : INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = lim / 2; s; s >>= 1) {
                        if (tid < s){
                            if(f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                                f_idx2reduce[tid] = f_idx2reduce[tid + s];
                        } 
                        itm.barrier();
                    }
                    int j1 = f_idx2reduce[0];
                    float_type low_value = -f_val2reduce[j1];

                    float_type local_diff = low_value - up_value;
                    if (numOfIter == 0) {
                        local_eps = sycl::max(eps, 0.1f * local_diff);
                        if (tid == 0) {
                            diff[0] = local_diff;
                        }
                    }

                    if (numOfIter > max_iter || local_diff < local_eps) {
                        alpha[wsi] = a;
                        alpha_diff[tid] = -(a - aold) * y;
                        diff[1] = numOfIter;
                        break;
                    }
                    itm.barrier();

                    //select j2 using second order heuristic
                    if (-up_value > -f && (is_I_low(a, y, Cp, Cn))) {
                        float_type aIJ = kd[i] + kd[tid] - 2 * kIwsI;
                        float_type bIJ = -up_value + f;
                        f_val2reduce[tid] = (-bIJ * bIJ / aIJ);
                    } else
                        f_val2reduce[tid] = INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = lim / 2; s; s >>= 1) {
                        if (tid < s){
                            if(f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                                f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        }
                        itm.barrier();
                    }
                    int j2 = f_idx2reduce[0];

                    //update alpha
                    if (tid == i)
                        *alpha_i_diff = y > 0 ? Cp - a : a;
                    if (tid == j2)
                        *alpha_j_diff = sycl::min(y > 0 ? a : Cn - a, (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
                    itm.barrier();
                    float_type l = sycl::min(*alpha_i_diff, *alpha_j_diff);

                    if (tid == i)
                        a += l * y;
                    if (tid == j2)
                        a -= l * y;

                    //update f
                    kernel_type kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                    f -= l * (kJ2wsI - kIwsI);
                    numOfIter++;
                }
            });
        }).wait();
        free(shared_mem, thunder::device_pool[0]);
    }
    void
    nu_smo_solve_kernel(const int *label, float_type *f_values, float_type *alpha, float_type *alpha_diff,
                        const int *working_set,
                        int ws_size, float C, const kernel_type *k_mat_rows, const kernel_type *k_mat_diag, int row_len,
                        float_type eps,
                        float_type *diff, int max_iter, size_t smem_size) {
        //"row_len" equals to the number of instances in the original training dataset.
        //allocate shared memory
        static size_t Glob = ws_size;
        static size_t Loca = ws_size; 
        int* shared_mem = (int *)malloc_device(Smem_size, thunder::device_pool[0]);
        thunder::device_pool[0].submit([&](handler &h){
            range<1> glob {Glob};
            range<1> loca {Loca};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm) {
                
                int *f_idx2reduce = shared_mem; //temporary memory for reduction
                float_type *f_val2reduce = (float_type *) &shared_mem[ws_size]; //f values used for reduction.
                float_type *alpha_i_diff = (float_type *) &shared_mem[ws_size + ws_size * sizeof(float_type) / sizeof(int)]; //delta alpha_i
                float_type *alpha_j_diff = &alpha_i_diff[1];
                kernel_type *kd = (kernel_type *) &alpha_j_diff[1]; // diagonal elements for kernel matrix

                //index, f value and alpha for each instance
                int tid = itm.get_global_id(0);
                int wsi = working_set[tid];
                kd[tid] = k_mat_diag[wsi];
                float_type y = label[wsi];
                float_type f = f_values[wsi];
                float_type a = alpha[wsi];
                float_type aold = a;
                itm.barrier();
                float_type local_eps;
                int numOfIter = 0;
                while (1) {
                    //select I_up (y=+1)
                    // if (y > 0 && a < C)
                    //     f_val2reduce[tid] = f;
                    // else
                    //     f_val2reduce[tid] = INFINITY;

                    f_val2reduce[tid] = (y > 0 && a < C) ? f : INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = itm.get_global_range(0) / 2; s; s >>= 1) {
                        if (tid < s && f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                            f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        itm.barrier();
                    }
                    int ip = f_idx2reduce[0];

                    float_type up_value_p = f_val2reduce[ip];
                    kernel_type kIpwsI = k_mat_rows[row_len * ip + wsi];//K[i, wsi]
                    itm.barrier();

                    //select I_up (y=-1)
                    // if (y < 0 && a > 0)
                    //     f_val2reduce[tid] = f;
                    // else
                    //     f_val2reduce[tid] = INFINITY;
                    f_val2reduce[tid] = (y < 0 && a > 0) ? f : INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = itm.get_global_range(0) / 2; s; s >>= 1) {
                        if (tid < s && f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                            f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        itm.barrier();
                    }
                    int in = f_idx2reduce[0];
                    float_type up_value_n = f_val2reduce[in];
                    kernel_type kInwsI = k_mat_rows[row_len * in + wsi];//K[i, wsi]
                    itm.barrier();

                    //select I_low (y=+1)
                    // if (y > 0 && a > 0)
                    //     f_val2reduce[tid] = -f;
                    // else
                    //     f_val2reduce[tid] = INFINITY;
                    f_val2reduce[tid] = (y > 0 && a > 0) ? -f : INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = itm.get_global_range(0) / 2; s; s >>= 1) {
                        if (tid < s && f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                            f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        itm.barrier();
                    }
                    int j1p = f_idx2reduce[0];
                    float_type low_value_p = -f_val2reduce[j1p];
                    itm.barrier();

                    //select I_low (y=-1)
                    if (y < 0 && a < C)
                        f_val2reduce[tid] = -f;
                    else
                        f_val2reduce[tid] = INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = itm.get_global_range(0) / 2; s; s >>= 1) {
                        if (tid < s && f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                            f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        itm.barrier();
                    }
                    int j1n = f_idx2reduce[0];
                    float_type low_value_n = -f_val2reduce[j1n];

                    float_type local_diff = sycl::max(low_value_p - up_value_p, low_value_n - up_value_n);

                    if (numOfIter == 0) {
                        local_eps = sycl::max(eps, 0.1 * local_diff);
                        if (tid == 0) {
                            diff[0] = local_diff;
                        }
                    }

                    if (numOfIter > max_iter || local_diff < local_eps) {
                        alpha[wsi] = a;
                        alpha_diff[tid] = -(a - aold) * y;
                        diff[1] = numOfIter;
                        break;
                    }
                    itm.barrier();

                    //select j2p using second order heuristic
                    if (-up_value_p > -f && y > 0 && a > 0) {
                        float_type aIJ = kd[ip] + kd[tid] - 2 * kIpwsI;
                        float_type bIJ = -up_value_p + f;
                        f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                    } else
                        f_val2reduce[tid] = INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = itm.get_global_range(0) / 2; s; s >>= 1) {
                        if (tid < s && f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                            f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        itm.barrier();
                    }
                    int j2p = f_idx2reduce[0];
                    float_type f_val_j2p = f_val2reduce[j2p];
                    itm.barrier();

                    //select j2n using second order heuristic
                    if (-up_value_n > -f && y < 0 && a < C) {
                        float_type aIJ = kd[in] + kd[tid] - 2 * kInwsI;
                        float_type bIJ = -up_value_n + f;
                        f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                    } else
                        f_val2reduce[tid] = INFINITY;
                    f_idx2reduce[tid] = tid;
                    itm.barrier();
                    for (int s = itm.get_global_range(0) / 2; s; s >>= 1) {
                        if (tid < s && f_val2reduce[f_idx2reduce[tid + s]] < f_val2reduce[f_idx2reduce[tid]])
                            f_idx2reduce[tid] = f_idx2reduce[tid + s]; 
                        itm.barrier();
                    }
                    int j2n = f_idx2reduce[0];

                    int i, j2;
                    float_type up_value;
                    kernel_type kIwsI;
                    if (f_val_j2p < f_val2reduce[j2n]) {
                        i = ip;
                        j2 = j2p;
                        up_value = up_value_p;
                        kIwsI = kIpwsI;
                    } else {
                        i = in;
                        j2 = j2n;
                        kIwsI = kInwsI;
                        up_value = up_value_n;
                    }
                    //update alpha
                    if (tid == i)
                        *alpha_i_diff = y > 0 ? C - a : a;
                    if (tid == j2)
                        *alpha_j_diff = sycl::min(y > 0 ? a : C - a, (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
                    itm.barrier();
                    float_type l = sycl::min(*alpha_i_diff, *alpha_j_diff);

                    if (tid == i)
                        a += l * y;
                    if (tid == j2)
                        a -= l * y;

                    //update f
                    kernel_type kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                    f -= l * (kJ2wsI - kIwsI);
                    numOfIter++;
                }
            });
        }).wait();
        free(shared_mem, thunder::device_pool[0]);
    }
    void
    c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
            SyncArray<float_type> &alpha_diff,
            const SyncArray<int> &working_set, float_type Cp, float_type Cn, const SyncArray<kernel_type> &k_mat_rows,
            const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
            int max_iter) {
        //LOG(INFO) << "c_smo_solve";
      //  c_smo.start();
        size_t ws_size = working_set.size();        
        
        size_t smem_size = 0;
        smem_size += ws_size * sizeof(int); //f_idx2reduce
        smem_size += ws_size * sizeof(float_type); //f_val2reduce
        smem_size += ws_size * sizeof(kernel_type); //kd
        smem_size += 2 * sizeof(float_type); //alpha diff
        c_smo_solve_kernel (y.device_data(), f_val.device_data(), alpha.device_data(), alpha_diff.device_data(),
                                working_set.device_data(), ws_size, Cp, Cn, k_mat_rows.device_data(), k_mat_diag.device_data(),
                                row_len, eps, diff.device_data(), max_iter, smem_size); 
     //   c_smo.end();
       
    }
    void nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                      SyncArray<float_type> &alpha_diff,
                      const SyncArray<int> &working_set, float_type C, const SyncArray<kernel_type> &k_mat_rows,
                      const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                      int max_iter) {
       // LOG(INFO) << "nu_smo_solve";
        size_t ws_size = working_set.size();
        size_t smem_size = 0;
        smem_size += ws_size * sizeof(int); //f_idx2reduce
        smem_size += ws_size * sizeof(float_type); //f_val2reduce
        smem_size += ws_size * sizeof(kernel_type); //kd
        smem_size += 2 * sizeof(float_type); //alpha diff
        nu_smo_solve_kernel (y.device_data(), f_val.device_data(), alpha.device_data(), alpha_diff.device_data(),
                                working_set.device_data(), ws_size, C, k_mat_rows.device_data(), k_mat_diag.device_data(),
                                row_len, eps, diff.device_data(), max_iter, smem_size);    
    }
    //to be optimized
    void
    update_f_kernel(float_type *f, int ws_size, const float_type *alpha_diff, const kernel_type *k_mat_rows,
                    int n_instances) {
        
        static size_t Glob = NUM_BLOCKS * BLOCK_SIZE; 
        static size_t Loca = BLOCK_SIZE;
        thunder::device_pool[0].submit([&](handler &h){
            range<1> glob {Glob};
            range<1> loca {Loca};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                int stp = itm.get_global_range(0);
                for (int idx = itm.get_global_id(0); idx < n_instances; idx += stp) {//one thread to update multiple fvalues.
                    double sum_diff = 0;
                    for (int i = 0; i < ws_size; ++i) {
                        double d = alpha_diff[i];
                        if (d != 0) {
                            sum_diff += d * k_mat_rows[i * n_instances + idx];
                        }
                    }
                    f[idx] -= sum_diff;
                }
            });
        }).wait();     
    }
    void
    update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff, const SyncArray<kernel_type> &k_mat_rows,
             int n_instances) {
        //LOG(INFO) << "update_f";
   //     upf.start();
        update_f_kernel(f.device_data(), alpha_diff.size(), alpha_diff.device_data(),
                        k_mat_rows.device_data(), n_instances);
       // upf.end();
    }
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
}