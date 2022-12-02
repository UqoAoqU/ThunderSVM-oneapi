/*
kernelmatrix_kernel with dpcpp
*/

#include <thundersvm/syncarray.h>
#include "thundersvm/kernel/kernelmatrix_kernel.h"
#include <CL/sycl.hpp>
#include <thundersvm/config.h>
#include "oneapi/mkl.hpp"
// #include <Eigen/Dense>
// #include <Eigen/Sparse>
#include "thundersvm/timer.h"
#include "thundersvm/device_pool.h"

using namespace oneapi;
using namespace sycl;
namespace svm_kernel {
    // timer get_wsi ("get_working_set_ins");
    // timer RBF ("RBF_kernel");
    // timer mul ("dns_csr_mul");
    // timer sum_kvalues ("sum_kernel_values");
    // timer poly("poly_kernel");
    // timer sigmoid("sigmoid_kernel");
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = 32 * 28;
    void 
    kernel_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                           kernel_type *data_rows,
                           int m, int n) {
#ifdef DEBUG
        LOG(INFO) << "get_working_set_ins";
#endif
        thunder::device_pool[0].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                for (int i = itm.get_global_id(0); i < m; i += BLOCK_SIZE * NUM_BLOCKS){
                    int row = data_row_idx[i];
                    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                        int col = col_ind[j];
                        //data_rows[col * m + i] = val[j]; // col-major for cuSPARSE 
                        data_rows[i * n + col] = val[j]; // row-major for Eigen
                    }
                }
            });
        }).wait();
    }
    void 
    multidevice_kernel_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                           kernel_type *data_rows,
                           int m, int n, int devi) {
#ifdef DEBUG
      //  LOG(INFO) << "get_working_set_ins";
#endif
        thunder::device_pool[devi].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                for (int i = itm.get_global_id(0); i < m; i += BLOCK_SIZE * NUM_BLOCKS){
                    int row = data_row_idx[i];
                    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                        int col = col_ind[j];
                        //data_rows[col * m + i] = val[j]; // col-major for cuSPARSE 
                        data_rows[i * n + col] = val[j]; // row-major for Eigen
                    }
                }
            });
        }).wait();
    }
    void
    kernel_RBF_kernel(const kernel_type *self_dot0, const kernel_type *self_dot1, kernel_type *dot_product, int m, int n,
                      kernel_type gamma) {
        //m rows of kernel matrix, where m is the working set size; n is the number of training instances
        thunder::device_pool[0].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                int stp = itm.get_global_range(0);
                int lim = m * n;
                for (int idx = itm.get_global_id(0); idx < lim; idx += BLOCK_SIZE * NUM_BLOCKS) {
                    int i = idx / n;//i is row id
                    int j = idx % n;//j is column id
                    dot_product[idx] = expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
                }
            });
        }).wait();
    }
    void
    kernel_RBF_kernel(const int *self_dot0_idx, const kernel_type *self_dot1, kernel_type *dot_product, int m, int n,
                      kernel_type gamma) {
        //compute m rows of kernel matrix, where m is the working set size and n is the number of training instances, according to idx
        thunder::device_pool[0].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                int stp = itm.get_global_range(0);
                int lim = m * n;
                for (int idx = itm.get_global_id(0); idx < lim; idx += BLOCK_SIZE * NUM_BLOCKS) {
                    int i = idx / n;//i is row id
                    int j = idx - i * n;//j is column id
                    dot_product[idx] = expf(-(self_dot1[self_dot0_idx[i]] + self_dot1[j] - dot_product[idx] * 2) * gamma);
                }
            });
        }).wait();
    }
    void
    kernel_sum_kernel_values(const float_type *coef, int total_sv, const int *sv_start, const int *sv_count,
                             const float_type *rho,
                             const kernel_type *k_mat, float_type *dec_values, int n_classes, int n_instances) {
        thunder::device_pool[0].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                int stp = itm.get_global_range(0);
                for (int idx = itm.get_global_id(0); idx < n_instances; idx += BLOCK_SIZE * NUM_BLOCKS) {
                    int k = 0;
                    int n_binary_models = n_classes * (n_classes - 1) / 2;
                    for (int i = 0; i < n_classes; ++i) {
                        for (int j = i + 1; j < n_classes; ++j) {
                            int si = sv_start[i];
                            int sj = sv_start[j];
                            int ci = sv_count[i];
                            int cj = sv_count[j];
                            const float_type *coef1 = &coef[(j - 1) * total_sv];
                            const float_type *coef2 = &coef[i * total_sv];
                            const kernel_type *k_values = &k_mat[idx * total_sv];
                            double sum = 0;
                            for (int l = 0; l < ci; ++l) {
                                sum += coef1[si + l] * k_values[si + l];
                            }
                            for (int l = 0; l < cj; ++l) {
                                sum += coef2[sj + l] * k_values[sj + l];
                            }
                            dec_values[idx * n_binary_models + k] = sum - rho[k];
                            k++;
                        }
                    }
                }
            });
        }).wait();
    }
    void
    kernel_poly_kernel(kernel_type *dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        thunder::device_pool[0].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                int stp = itm.get_global_range(0);
                for (int idx = itm.get_global_id(0); idx < mn; idx += BLOCK_SIZE * NUM_BLOCKS) {
                    dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
                }
            });
        }).wait();
    }
    void kernel_sigmoid_kernel(kernel_type *dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        thunder::device_pool[0].submit([&](handler &h){
            range glob {NUM_BLOCKS * BLOCK_SIZE};
            range loca {BLOCK_SIZE};
            h.parallel_for(nd_range{glob, loca}, [=](nd_item<1> itm){
                int stp = itm.get_global_range(0);
                for (int idx = itm.get_global_id(0); idx < mn; idx += BLOCK_SIZE * NUM_BLOCKS) {
                    dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
                }
            });
        }).wait();
    }

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances) {
        //LOG(INFO) << "sum_kernel_values";
       // sum_kvalues.start();
        kernel_sum_kernel_values(coef.device_data(), total_sv, sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), k_mat.device_data(), dec_values.device_data(),
                           n_classes, n_instances);
       // sum_kvalues.end();
    }
    void
    get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr,
                        const int*data_row_idx, kernel_type* data_rows, int m, int n) {
        //LOG(INFO) << "get_workiung_set_ins";
       // get_wsi.start();
        kernel_get_working_set_ins(val, col_ind, row_ptr, data_row_idx, data_rows, m, n);
       // get_wsi.end();
    }
    void 
    multidevice_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr,
                        const int*data_row_idx, kernel_type* data_rows, int m, int n, int devi) {
        //LOG(INFO) << "get_workiung_set_ins";
    //    get_wsi.start();
        multidevice_kernel_get_working_set_ins(val, col_ind, row_ptr, data_row_idx, data_rows, m, n, devi);
      //  get_wsi.end();
    }
    void
    RBF_kernel(const SyncArray<kernel_type> &self_dot0, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n,
               kernel_type gamma) {
        //LOG(INFO) << "RBF_kernel";
    //    RBF.start();
        kernel_RBF_kernel(self_dot0.device_data(), self_dot1.device_data(),
                           dot_product.device_data(), m, n, gamma);
      //  RBF.end();
    }

    void
    RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        //LOG(INFO) << "RBF_kernel";
     //   RBF.start();
        kernel_RBF_kernel(self_dot0_idx.device_data(), self_dot1.device_data(),
                           dot_product.device_data(), m, n, gamma);
       // RBF.end();
    }

    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        //LOG(INFO) << "poly_kernel";
    //    poly.start();
        kernel_poly_kernel(dot_product.device_data(), gamma, coef0, degree, mn);
       // poly.end();
    }

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        //LOG(INFO) << "sigmoid_kernel";
     //   sigmoid.start();
        kernel_sigmoid_kernel(dot_product.device_data(), gamma, coef0, mn);
      //  sigmoid.end();
    }

 
    bool mklsparse_init;
    void dns_csr_mul(int m, int n, int k, SyncArray<kernel_type> &dense_mat, SyncArray<kernel_type> &csr_val,
                    SyncArray<int> &csr_row_ptr, SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result) {         
#ifdef DEBUG
        LOG(INFO) << "dns_csr_mul";
#endif
        mkl::sparse::matrix_handle_t handle;
        try{
   
            mkl::sparse::init_matrix_handle(&handle);    
          //  mul.start();
            const kernel_type one(1);
            const kernel_type zero(0);
            //LOG(INFO) << "mkl::sparse::gemm(thunder::device_pool[0], mkl::layout::row_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle,dense_mat.device_data(), n, k,zero, result.device_data(), m)";
            mkl::sparse::set_csr_data(handle, m, k, mkl::index_base::zero,
                                    csr_row_ptr.device_data(), csr_col_ind.device_data(), csr_val.device_data());
           // LOG(INFO) << "start gemm";
           // LOG(INFO) << "total memory usage:" << SyncMem::get_total_memory_size() / 1024.0 / 1024.0 <<"MB";
            
            thunder::device_pool[0].wait();
       //     mul.end();
            auto e = mkl::sparse::gemm(thunder::device_pool[0], mkl::layout::col_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle, 
                                dense_mat.device_data(), n, k, 
                                zero, result.device_data(), m);
                //e.get_info<sycl::info::event::command_execution_status>();
            //LOG(INFO) << "finish gemm";      
          
            mkl::sparse::release_matrix_handle(&handle, {e});
        }
        catch (sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;

            oneapi::mkl::sparse::release_matrix_handle(&handle);

            return;
        }
        catch (std::exception const &e) {
            std::cout << "\t\tCaught std exception:\n" << e.what() << std::endl;
            oneapi::mkl::sparse::release_matrix_handle(&handle);

            return;
        }
  
    }
    void multidevice_dns_csr_mul(int m, int n, int k, kernel_type* dense_mat, kernel_type* csr_val,
                    int* csr_row_ptr, int* csr_col_ind, int nnz,
                    kernel_type* result, int devi) {         
 #ifdef DEBUG
        // LOG(INFO) << "dns_csr_mul";
 #endif
        mkl::sparse::matrix_handle_t handle;
        try{
   
            mkl::sparse::init_matrix_handle(&handle);    
        //    mul.start();
            const kernel_type one(1);
            const kernel_type zero(0);
            //LOG(INFO) << "mkl::sparse::gemm(thunder::device_pool[0], mkl::layout::row_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle,dense_mat.device_data(), n, k,zero, result.device_data(), m)";
            mkl::sparse::set_csr_data(handle, m, k, mkl::index_base::zero,
                                    csr_row_ptr, csr_col_ind, csr_val);
           // LOG(INFO) << "start gemm";
           // LOG(INFO) << "total memory usage:" << SyncMem::get_total_memory_size() / 1024.0 / 1024.0 <<"MB";
            
            thunder::device_pool[devi].wait();
#ifdef DEBUG
        //LOG(INFO) << csr_val << " " << csr_row_ptr << " " << csr_col_ind << " " << result;
#endif
            auto e = mkl::sparse::gemm(thunder::device_pool[devi], mkl::layout::col_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle, 
                                dense_mat, n, k, 
                                zero, result, m);
                //e.get_info<sycl::info::event::command_execution_status>();
            //LOG(INFO) << "finish gemm";      
        //    mul.end();
            mkl::sparse::release_matrix_handle(&handle, {e});
        }
        catch (sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;

            oneapi::mkl::sparse::release_matrix_handle(&handle);

            return;
        }
        catch (std::exception const &e) {
            std::cout << "\t\tCaught std exception:\n" << e.what() << std::endl;
            oneapi::mkl::sparse::release_matrix_handle(&handle);

            return;
        }
  
  
    }
    void multidevice_dns_csr_mul(int m, int n, int k, kernel_type* dense_mat, kernel_type* result, int devi,
                                mkl::sparse::matrix_handle_t &handle) {
        const kernel_type one(1);
        const kernel_type zero(0);
        mkl::sparse::gemm(thunder::device_pool[devi], mkl::layout::col_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle, 
                                dense_mat, n, k, 
                                zero, result, m).wait();
    }
    void multidevice_div_dns_csr_mul(int m, int n, int k, kernel_type* dense_mat, kernel_type** csr_val,
                    int** csr_row_ptr, int** csr_col_ind, int nnz,
                    kernel_type* result, int devi) {         
#ifdef DEBUG
         LOG(INFO) << "dns_csr_mul";
#endif
        mkl::sparse::matrix_handle_t handle[2];
        try{
   
            mkl::sparse::init_matrix_handle(&handle[0]);
            mkl::sparse::init_matrix_handle(&handle[1]);    
         //   mul.start();
            const kernel_type one(1);
            const kernel_type zero(0);
            //LOG(INFO) << "mkl::sparse::gemm(thunder::device_pool[0], mkl::layout::row_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle,dense_mat.device_data(), n, k,zero, result.device_data(), m)";
            mkl::sparse::set_csr_data(handle[0], m, k, mkl::index_base::zero,
                                    csr_row_ptr[0], csr_col_ind[0], csr_val[0]);
            mkl::sparse::set_csr_data(handle[1], m, k, mkl::index_base::zero,
                                    csr_row_ptr[1], csr_col_ind[1], csr_val[1]);                    
           // LOG(INFO) << "start gemm";
           // LOG(INFO) << "total memory usage:" << SyncMem::get_total_memory_size() / 1024.0 / 1024.0 <<"MB";
            
            //thunder::device_pool[devi].wait();
//#pragma parallel omp for num_threads(2)
            for (int i = 0; i < 2; ++i) {
                mkl::sparse::gemm(thunder::device_pool[devi], mkl::layout::col_major, mkl::transpose::nontrans, mkl::transpose::nontrans, one, handle[i], 
                                dense_mat + (n * k / 2) * i, n / 2, k, 
                                zero, result + (m * n / 2) * i, m);

            }
                //e.get_info<sycl::info::event::command_execution_status>();
#ifdef DEBUG     
            LOG(INFO) << "finish gemm";
#endif      
            thunder::device_pool[devi].wait();
         //   mul.end();
            mkl::sparse::release_matrix_handle(&handle[0]);
            mkl::sparse::release_matrix_handle(&handle[1]);
        }
        catch (sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;

            oneapi::mkl::sparse::release_matrix_handle(&handle[0]);
            oneapi::mkl::sparse::release_matrix_handle(&handle[1]);
            return;
        }
        catch (std::exception const &e) {
            std::cout << "\t\tCaught std exception:\n" << e.what() << std::endl;
            oneapi::mkl::sparse::release_matrix_handle(&handle[0]);
            oneapi::mkl::sparse::release_matrix_handle(&handle[1]);

            return;
        }
  
  
    }
}



