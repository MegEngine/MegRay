#include "communicator.h"

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status ShmCommunicator::scatter(const void* sendbuff, void* recvbuff,
                                size_t recvlen, DType dtype, uint32_t root,
                                std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    auto buf_size = recvlen * get_dtype_size(dtype);
    auto stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    void* shmadd = alloc_shm(buf_size * m_nranks);
    m_client->barrier();
    if (root == m_rank) {
        CUDA_ASSERT(cudaMemcpyAsync(shmadd, sendbuff, buf_size * m_nranks,
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_ASSERT(cudaStreamSynchronize(stream));
    }
    m_client->barrier();
    CUDA_ASSERT(cudaMemcpyAsync(recvbuff, (uint8_t*)shmadd + buf_size * m_rank,
                                buf_size, cudaMemcpyHostToDevice, stream));
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    m_client->barrier();
    return MEGRAY_OK;
}

}  // namespace MegRay
