#ifndef DEVICE_POOL_H
#define DEVICE_POOL_H
#include <CL/sycl.hpp>
#include <vector>

#ifndef USE_DPCPP
#define USE_DPCPP
#endif
//#define DEBUG


namespace thunder {
    // device[0] for storage,
    // device[0~3] for computation
    extern std::vector<sycl::queue>device_pool;
    void init_device();
};

#endif