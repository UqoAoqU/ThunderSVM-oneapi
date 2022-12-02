#include "thundersvm/device_pool.h"
#include <thundersvm/util/log.h>
using namespace sycl;

namespace thunder {
    std::vector<sycl::queue>device_pool;
    void init_device(){    
        auto exception_handler = [](cl::sycl::exception_list exceptions) {
            for (std::exception_ptr const &e : exceptions) {
                try {
                    std::rethrow_exception(e);
                }
                catch (cl::sycl::exception const &e) {
                    LOG(FATAL) << "Caught asynchronous SYCL "
                                "exception during sparse::gemm:\n"
                            << e.what() << std::endl;
                }
            }
        };
        auto P = platform(gpu_selector{});
        auto RootDevices = P.get_devices();
        for (auto &D : RootDevices) {
            auto SubDevices = D.create_sub_devices<
            cl::sycl::info::partition_property::partition_by_affinity_domain>(
            cl::sycl::info::partition_affinity_domain::next_partitionable);
            for(auto &subD : SubDevices) {
                auto Q = queue(subD, exception_handler);
                device_pool.push_back(Q);
            }
        }
        LOG(INFO) << "Number of GPU tiles found = " << device_pool.size();
    }
}