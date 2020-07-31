#include "megray/core/context.h"

#ifdef MEGRAY_WITH_CUDA

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                \
    do {                                                \
        cudaError_t status = (expr);                    \
        if (status != cudaSuccess) {                    \
            MEGRAY_ERROR("cuda error [%d]: %s", status, \
                cudaGetErrorString(status));            \
            return MEGRAY_CUDA_ERR;                     \
        }                                               \
    } while (0)

#define CUDA_ASSERT(expr)                               \
    do {                                                \
        cudaError_t status = (expr);                    \
        if (status != cudaSuccess) {                    \
            MEGRAY_ERROR("cuda error [%d]: %s", status, \
                cudaGetErrorString(status));            \
            MEGRAY_THROW("cuda error");                 \
        }                                               \
    } while (0)

namespace MegRay{

class CudaContext: public Context{
public:
    CudaContext(cudaStream_t stream): m_stream{stream}{}
    ContextType type() const override{
        return MEGRAY_CTX_CUDA;
    }
    cudaStream_t get_stream(){
        return m_stream;
    }
private:
    cudaStream_t m_stream;
};

}

#endif