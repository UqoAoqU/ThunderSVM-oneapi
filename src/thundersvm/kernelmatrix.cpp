//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/svmparam.h>
#include "thundersvm/kernelmatrix.h"
#include "thundersvm/kernel/kernelmatrix_kernel.h"
#include <CL/sycl.hpp>
using namespace sycl;
using namespace svm_kernel;
using thunder::device_pool;
using namespace oneapi;
KernelMatrix::KernelMatrix(const DataSet::node2d &instances, SvmParam param) {
    n_instances_ = instances.size();
    n_features_ = 0;
    this->param = param;

    //three arrays for csr representation
    vector<kernel_type> csr_val;
    vector<int> csr_col_ind;//index of each value of all the instances
    vector<int> csr_row_ptr(1, 0);//the start positions of the instances

    vector<kernel_type> csr_self_dot;
    for (int i = 0; i < n_instances_; ++i) {//convert libsvm format to csr format
        float_type self_dot = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            csr_val.push_back(instances[i][j].value);
            self_dot += instances[i][j].value * instances[i][j].value;
            csr_col_ind.push_back(instances[i][j].index);//libSVM data format is one-based, convert to zero-based
            if (instances[i][j].index > n_features_) n_features_ = instances[i][j].index;
        }
        csr_row_ptr.push_back(csr_row_ptr.back() + instances[i].size());
        csr_self_dot.push_back(self_dot);
    }
    n_features_++;

    //three arrays (on GPU/CPU) for csr representation
    val_.resize(csr_val.size());
    col_ind_.resize(csr_col_ind.size());
    row_ptr_.resize(csr_row_ptr.size());
    //copy data to the three arrays
    val_.copy_from(csr_val.data(), val_.size());
    col_ind_.copy_from(csr_col_ind.data(), col_ind_.size());
    row_ptr_.copy_from(csr_row_ptr.data(), row_ptr_.size());

    val_c = new kernel_type** [device_pool.size()];
    row_ptr_c = new int** [device_pool.size()];
    col_ind_c = new int** [device_pool.size()];
    dns_data_c = new kernel_type* [device_pool.size()]{0};
    idx_c = new int* [device_pool.size()] {0};
    Re_c = new kernel_type* [device_pool.size()]{0};
    handle = new oneapi::mkl::sparse::matrix_handle_t [device_pool.size()];

    for (int i = 0; i < device_pool.size(); ++i) {
        val_c[i] = new kernel_type *[2]{0};
        col_ind_c[i] = new int *[2]{0};
        row_ptr_c[i] = new int *[2]{0};
    }    
    //pay attension to the initailization
    val_c[0][0] = val_.device_data();
    row_ptr_c[0][0] = row_ptr_.device_data();
    col_ind_c[0][0] = col_ind_.device_data();
    for (int i = 0; i < device_pool.size(); ++i)
        mkl::sparse::init_matrix_handle(&handle[i]);
    for (int i = 1; i < device_pool.size(); ++i) {
        val_c[i][0] = malloc_device<kernel_type>(val_.size(), device_pool[i]);
        row_ptr_c[i][0] = malloc_device<int>(row_ptr_.size(), device_pool[i]);
        col_ind_c[i][0] = malloc_device<int>(col_ind_.size(), device_pool[i]);
    }
    mem_size = 4 * (val_.mem_size() + row_ptr_.mem_size() + col_ind_.mem_size());
    //get the extra copy
    div = 0;
    init_flag = 0;

    self_dot_.resize(n_instances_);
    self_dot_.copy_from(csr_self_dot.data(), self_dot_.size());

    nnz_ = csr_val.size();//number of nonzero

    //pre-compute diagonal elements

    diag_.resize(n_instances_);
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            for (int i = 0; i < n_instances_; ++i) {
                diag_.host_data()[i] = 1;//rbf kernel
            }
            break;
        case SvmParam::LINEAR:
            diag_.copy_from(self_dot_);
            break;
        case SvmParam::POLY:
            diag_.copy_from(self_dot_);
            poly_kernel(diag_, param.gamma, param.coef0, param.degree, diag_.size());
            break;
        case SvmParam::SIGMOID:
            diag_.copy_from(self_dot_);
            sigmoid_kernel(diag_, param.gamma, param.coef0, diag_.size());
        default:
            break;
    }

}
KernelMatrix::~KernelMatrix() {
    for (int i = 0; i < device_pool.size(); ++i) {
        device_pool[i].wait();
        mkl::sparse::release_matrix_handle(&handle[i]);
        if (i != 0 ){
            free(val_c[i][0], device_pool[i]);
            free(row_ptr_c[i][0], device_pool[i]);
            free(col_ind_c[i][0], device_pool[i]);
        }
        if(dns_data_c[i])
            free(dns_data_c[i],device_pool[i]);
        if (idx_c[i])
            free(idx_c[i],device_pool[i]);
        if (Re_c[i])
            free(Re_c[i], device_pool[i]);
    }
    for (int i = 0; i < device_pool.size(); ++i) {
        delete[] val_c[i];
        delete[] row_ptr_c[i];
        delete[] col_ind_c[i];
    }
    if (val_c) delete[] val_c;
    if (row_ptr_c) delete[] row_ptr_c;
    if (col_ind_c) delete[] col_ind_c;
    if (dns_data_c) delete[] dns_data_c;
    if (Re_c) delete[] Re_c;
    if (handle) delete[] handle;
}
void KernelMatrix::get_rows(const SyncArray<int> &idx,
                            SyncArray<kernel_type> &kernel_rows)  {//compute multiple rows of kernel matrix according to idx
    CHECK_GE(kernel_rows.size(), idx.size() * n_instances_) << "kernel_rows memory is too small";
#ifdef USE_DPCPP
    get_dot_product_dns_csr(idx, kernel_rows);
#else
	if(n_features_ < 1000000)
		get_dot_product_dns_csr(idx, kernel_rows);
	else
		get_dot_product_csr_csr(idx, kernel_rows);
//    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            RBF_kernel(idx, self_dot_, kernel_rows, idx.size(), n_instances_, param.gamma);
			break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
}

void KernelMatrix::get_rows(const DataSet::node2d &instances,
                            SyncArray<kernel_type> &kernel_rows)  {//compute the whole (sub-) kernel matrix of the given instances.
    CHECK_GE(kernel_rows.size(), instances.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(instances, kernel_rows);

    //compute self dot
    //TODO use thrust
    SyncArray<kernel_type> self_dot(instances.size());
    kernel_type *self_dot_data = self_dot.host_data(); 
    for (int i = 0; i < instances.size(); ++i) {
        kernel_type sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            sum += instances[i][j].value * instances[i][j].value;
        }
        self_dot_data[i] = sum;
    }
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            RBF_kernel(self_dot, self_dot_, kernel_rows, instances.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
}
#ifdef USE_DPCPP
void KernelMatrix::init_devices(const int & ws_size) {
    init_flag = 1;
    int poolsize = device_pool.size();
//#pragma omp parallel for
    for (int i = 0; i < poolsize; ++i) {
        Re_c[i] = malloc_device<kernel_type>(ws_size / poolsize * n_instances_, device_pool[i]); 
        dns_data_c[i] = malloc_device<kernel_type>(ws_size / poolsize * n_features_, device_pool[i]);
        device_pool[i].memset(dns_data_c[i], 0, sizeof(kernel_type) * (ws_size) / poolsize * n_features_);
        idx_c[i] = malloc_device<int> (ws_size / poolsize, device_pool[i]);
        if (i) {
            device_pool[i].memcpy(val_c[i][0], val_.host_data(), val_.mem_size());
            device_pool[i].memcpy(row_ptr_c[i][0], row_ptr_.host_data(), row_ptr_.mem_size());
            device_pool[i].memcpy(col_ind_c[i][0], col_ind_.host_data(), col_ind_.mem_size());
        }
        else {
            val_c[i][0] = val_.device_data();
            row_ptr_c[i][0] = row_ptr_.device_data();
            col_ind_c[i][0] = col_ind_.device_data();
        }
        mkl::sparse::set_csr_data(handle[i], n_instances_,  n_features_, mkl::index_base::zero, 
                                row_ptr_c[i][0], col_ind_c[i][0], val_c[i][0]);
    }   
    for (int i = 0; i < poolsize; ++i)
        device_pool[i].wait();
    mem_size += (ws_size * (n_features_ + n_instances_));
}
#endif
const SyncArray<kernel_type> &KernelMatrix::diag() const {
    return this->diag_;
}



void
KernelMatrix::dns_csr_mul(SyncArray<kernel_type> &dense_mat, int n_rows, SyncArray<kernel_type> &result) {
    CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
    svm_kernel::dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, val_, row_ptr_, col_ind_, nnz_, result);
}
void
KernelMatrix::multidevice_dns_csr_mul(kernel_type* dense_mat, int n_rows, kernel_type* result,int devi, int div) {
    //CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
        //svm_kernel::multidevice_dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, val_c[devi][0], row_ptr_c[devi][0], col_ind_c[devi][0], nnz_, result, devi);
    svm_kernel::multidevice_dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, result, devi, handle[devi]);


}
#ifndef USE_DPCPP
void
KernelMatrix::csr_csr_mul(const SyncArray<kernel_type> &ws_val, int n_rows, const SyncArray<int> &ws_col_ind,
                          const SyncArray<int> &ws_row_ptr, SyncArray<kernel_type> &result) const {
    svm_kernel::csr_csr_mul(n_instances_, n_rows, n_features_, ws_val, ws_col_ind, ws_row_ptr,
                            val_, row_ptr_, col_ind_, nnz_, ws_val.size(), result);
}

void
KernelMatrix::dns_dns_mul(const SyncArray<kernel_type> &dense_mat, int n_rows,
                          const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result) const {
    CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
    svm_kernel::dns_dns_mul(n_instances_, n_rows, n_features_, dense_mat, origin_dense, result);
}
#endif
void KernelMatrix::get_dot_product_dns_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) {

    if (init_flag == 1) {
       // LOG(INFO) << "get_dot_product_dns_csr";
        int poolsize = device_pool.size();
        if (idx.size () % poolsize != 0) {
            LOG(FATAL) <<"ERRRRRRR";
        }
        //???
        size_t ws = idx.size() / poolsize, dot_product_mem_size = dot_product.mem_size() / poolsize;
        size_t dot_product_size = dot_product.size() / poolsize;
        auto host_dot_product = dot_product.host_data();
        const int *host_idx = idx.host_data();
    
#pragma omp parallel for num_threads(4)
        for (int i = 0; i < poolsize; ++i) {
            device_pool[i].memcpy(idx_c[i], host_idx + ws * i, ws * sizeof(int));
            device_pool[i].wait();
            multidevice_get_working_set_ins(val_c[i][0], col_ind_c[i][0], row_ptr_c[i][0], idx_c[i], dns_data_c[i], ws, n_features_, i);
        }
#pragma omp parallel for num_threads(4)
        for (int i = 0; i < poolsize; ++i){
            multidevice_dns_csr_mul(dns_data_c[i], ws, Re_c[i], i, div);
            device_pool[i].memcpy(host_dot_product + dot_product_size * i, Re_c[i], dot_product_mem_size).wait();
        }
        for (int i = 0; i < poolsize; ++i){
            device_pool[i].memset(dns_data_c[i], 0, sizeof(kernel_type) * ws * n_features_);
            //device_pool[i].memset(Re_c[i], 0, sizeof(kernel_type) * ws * n_instances_);
        }
    }
    else {
        SyncArray<kernel_type> data_rows(idx.size() * n_features_);
        data_rows.mem_set(0);
        //dpcpp -> device 
        
        get_working_set_ins(val_.device_data(), col_ind_.device_data(), row_ptr_.device_data(), idx.device_data(), data_rows.device_data(), idx.size(), n_features_);
#ifdef DEBUG
        // LOG(INFO) << "[KERNEL]" << "check the data_rows";
        // data_rows.log(std::cout);
        // std::cout << '\n';
        // col_ind_.log(std::cout);
        // std::cout << '\n';
        // row_ptr_.log(std::cout);
        // std::cout << '\n';
        // idx.log(std::cout);
        // std::cout << '\n';
#endif
        dns_csr_mul(data_rows, idx.size(), dot_product);
    }
}

void KernelMatrix::get_dot_product(const DataSet::node2d &instances, SyncArray<kernel_type> &dot_product){
    SyncArray<kernel_type> dense_ins(instances.size() * n_features_);
    dense_ins.mem_set(0);
    kernel_type *dense_ins_data = dense_ins.host_data();
    for (int i = 0; i < instances.size(); ++i) {
        kernel_type sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            if (instances[i][j].index < n_features_) {
                //col major for cuSPARSE, row major for Eigen
#ifdef USE_DPCPP
                dense_ins_data[instances[i][j].index * instances.size() + i] = instances[i][j].value;
#else
                dense_ins_data[i * n_features_ + instances[i][j].index] = instances[i][j].value;
#endif
                sum += instances[i][j].value * instances[i][j].value;
            } else {
//                LOG(WARNING)<<"the number of features in testing set is larger than training set";
            }
        }
    }
    dns_csr_mul(dense_ins, instances.size(), dot_product);
}
#ifndef USE_DPCPP
void KernelMatrix::get_dot_product_csr_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const {
    SyncArray<kernel_type> ws_val;
    SyncArray<int> ws_col_ind;
    SyncArray<int> ws_row_ptr;
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, ws_val, ws_col_ind, ws_row_ptr, idx.size());
    csr_csr_mul(ws_val, idx.size(), ws_col_ind, ws_row_ptr, dot_product);
}

void KernelMatrix::get_dot_product_dns_dns(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const {
    SyncArray<kernel_type> data_rows(idx.size() * n_features_);
    data_rows.mem_set(0);
    SyncArray<kernel_type> origin_dense(n_instances_ * n_features());
    origin_dense.mem_set(0);
    SyncArray<int> origin_idx(n_instances_);
    int *origin_idx_data = origin_idx.host_data();
    for (int i = 0; i < n_instances_; ++i) {
        origin_idx_data[i] = i;
    }
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size(), n_features_);
    get_working_set_ins(val_, col_ind_, row_ptr_, origin_idx, origin_dense, origin_idx.size(), n_features_);
    dns_dns_mul(data_rows, idx.size(), origin_dense, dot_product);
}
#endif
