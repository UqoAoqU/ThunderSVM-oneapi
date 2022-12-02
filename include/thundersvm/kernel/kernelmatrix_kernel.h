//
// Created by jiashuai on 17-9-20.
//

#ifndef THUNDERSVM_KERNELMATRIX_KERNEL_H
#define THUNDERSVM_KERNELMATRIX_KERNEL_H

#include "thundersvm/thundersvm.h"
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncarray.h>
#include <oneapi/mkl.hpp>
namespace svm_kernel {
    void
    get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr,
                        const int*data_row_idx, kernel_type* data_rows, int m, int n);  
    void 
    multidevice_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr,
                        const int*data_row_idx, kernel_type* data_rows, int m, int n, int devi);
    void
    get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                        const SyncArray<int> &data_row_idx, SyncArray <kernel_type>& ws_val,
                        SyncArray<int> &ws_col_ind, SyncArray<int> &ws_row_ptr, int m);
  
    void
    RBF_kernel(const SyncArray<kernel_type> &self_dot0, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n,
               kernel_type gamma);

    void
    RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma);

    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn);

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn);

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances);

    void dns_csr_mul(int m, int n, int k, SyncArray<kernel_type> &dense_mat, SyncArray<kernel_type> &csr_val,
                     SyncArray<int> &csr_row_ptr, SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result);
    
    void multidevice_dns_csr_mul(int m, int n, int k, kernel_type* dense_mat, kernel_type* csr_val,
                    int* csr_row_ptr, int* csr_col_ind, int nnz,
                    kernel_type* result, int devi);
    void multidevice_dns_csr_mul(int m, int n, int k, kernel_type* dense_mat, kernel_type* result, int devi,
                                oneapi::mkl::sparse::matrix_handle_t & handle);
    void multidevice_div_dns_csr_mul(int m, int n, int k, kernel_type* dense_mat, kernel_type** csr_val,
                    int** csr_row_ptr, int** csr_col_ind, int nnz,
                    kernel_type* result, int devi);
#ifndef USE_DPCPP
    void csr_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &ws_val, const SyncArray<int> &ws_col_ind,
                     const SyncArray<int> &ws_row_ptr, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz, int nnz2,
                     SyncArray<kernel_type> &result);

    void dns_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat,
                     const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result);
    
#endif
}
#endif //THUNDERSVM_KERNELMATRIX_KERNEL_H

