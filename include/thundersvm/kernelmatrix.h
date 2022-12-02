//
// Created by jiashuai on 17-9-19.
//

#ifndef THUNDERSVM_KERNELMATRIX_H
#define THUNDERSVM_KERNELMATRIX_H

#include "thundersvm.h"
#include "syncarray.h"
#include "dataset.h"
#include "svmparam.h"
#include "oneapi/mkl.hpp"

/**
 * @brief The management class of kernel values.
 */
class KernelMatrix{
public:
    /**
     * Create KernelMatrix with given instances (training data or support vectors).
     * @param instances the instances, either are training instances for training, or are support vectors for prediction.
     * @param param kernel_type in parm is used
     */
    explicit KernelMatrix(const DataSet::node2d &instances, SvmParam param);
    ~KernelMatrix();
    /**
     * return specific rows in kernel matrix
     * @param [in] idx the indices of the rows
     * @param [out] kernel_rows
     */
    void get_rows(const SyncArray<int> &idx, SyncArray<kernel_type> &kernel_rows);

    /**
     * return kernel values of each given instance and each instance stored in KernelMatrix
     * @param [in] instances the given instances
     * @param [out] kernel_rows
     */
    void get_rows(const DataSet::node2d &instances, SyncArray<kernel_type> &kernel_rows);
#ifdef USE_DPCPP
    //copy the data to multiple devices.
    void init_devices(const int &ws_size);
    //initialize the matrix handle.
    void set_csr_data();
    //release the handle
//   void release_handle();
#endif
    ///return the diagonal elements of kernel matrix
    const SyncArray<kernel_type> &diag() const;

    ///the number of instances in KernelMatrix
    size_t n_instances() const { return n_instances_; };

    ///the maximum number of features of instances
    size_t n_features() const { return n_features_; }

    ///the number of non-zero features of all instances
    size_t nnz() const {return nnz_;};//number of nonzero
    //
    size_t get_mem_size() const {return mem_size;}

    size_t get_div()const {return div;}
private:
    KernelMatrix &operator=(const KernelMatrix &) const;

    KernelMatrix(const KernelMatrix &);

    SyncArray<kernel_type> val_;
    SyncArray<int> col_ind_;
    SyncArray<int> row_ptr_;
    SyncArray<kernel_type> diag_;
    SyncArray<kernel_type> self_dot_;
    size_t nnz_;
    size_t n_instances_;
    size_t n_features_;
    SvmParam param;

    //the copies of the data.
    int ***col_ind_c, ***row_ptr_c, **idx_c;
    kernel_type ***val_c, **dns_data_c, **Re_c;
    size_t mem_size;
    bool init_flag, div;
    oneapi::mkl::sparse::matrix_handle_t *handle;
    
    void dns_csr_mul(SyncArray<kernel_type> &dense_mat, int n_rows, SyncArray<kernel_type> &result);

    void multidevice_dns_csr_mul(kernel_type* dense_mat, int n_rows, kernel_type* result, int devi, int Div);
#ifndef USE_DPCPP
    void csr_csr_mul(const SyncArray<kernel_type> &ws_val, int n_rows, const SyncArray<int> &ws_col_ind,
                              const SyncArray<int> &ws_row_ptr, SyncArray<kernel_type> &result) const;
    void dns_dns_mul(const SyncArray<kernel_type> &dense_mat, int n_rows,
                              const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result) const;
#endif
    void get_dot_product_dns_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product);

    void get_dot_product_csr_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;

    void get_dot_product_dns_dns(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;

    void get_dot_product(const DataSet::node2d &instances, SyncArray<kernel_type> &dot_product);

    void get_dot_product_sparse(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;
};
#endif //THUNDERSVM_KERNELMATRIX_H
