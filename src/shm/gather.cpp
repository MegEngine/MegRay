#include "communicator.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status ShmCommunicator::_shm_gather(const void* sendbuff, void* recvbuff,
                                    size_t sendlen, DType dtype, uint32_t root,
                                    std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    auto stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    auto size = get_dtype_size(dtype);
    auto buff_size = size * sendlen;
    void* shmadd = alloc_shm(buff_size * m_nranks + m_nranks * sizeof(int));
    volatile int* mutex =
            (volatile int*)((uint8_t*)shmadd + buff_size * m_nranks);
    for (int i = 0; i < m_nranks; i++) {
        mutex[i] = 0;
    }
    m_client->barrier();
    CUDA_CHECK(cudaMemcpyAsync((uint8_t*)shmadd + buff_size * m_rank,
                               (uint8_t*)sendbuff, buff_size,
                               cudaMemcpyDeviceToHost, stream_vec[m_rank]));
    CUDA_CHECK(cudaStreamSynchronize(stream_vec[m_rank]));
    mutex[m_rank] = 1;  // set the flag with done
    if (m_rank == root) {
        _shm_barrier_sum(mutex);  // block all process
        CUDA_CHECK(cudaMemcpyAsync((uint8_t*)recvbuff, (uint8_t*)shmadd,
                                   buff_size * m_nranks, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    m_client->barrier();
    return MEGRAY_OK;
}

Status ShmCommunicator::gather(const void* sendbuff, void* recvbuff,
                               size_t sendlen, DType dtype, uint32_t root,
                               std::shared_ptr<Context> ctx) {
    if (!m_is_single_machine) {
        MEGRAY_THROW("shm gather only support in single machine now");
    }
    return _shm_gather(sendbuff, recvbuff, sendlen, dtype, root, ctx);
}
}  // namespace MegRay
