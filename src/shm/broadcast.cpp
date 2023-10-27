#include "communicator.h"
#include "megray/cuda_context.h"
#include "utils.h"

namespace MegRay {

Status ShmCommunicator ::_shm_broadcast(const void* sendbuff, void* recvbuff,
                                        size_t len, DType dtype, uint32_t root,
                                        std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t size = get_dtype_size(dtype);
    size_t buff_size = len * size;
    // the one represents the bool bit
    // when equals 1 represent done
    void* shmadd = alloc_shm(buff_size + sizeof(int));
    if (root == m_rank) {
        // copy the root buffer to shared memory
        // also the device memory
        // assure cuda copy is done and then set mutex = 1
        CUDA_CHECK(cudaMemcpyAsync(shmadd, sendbuff, buff_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(recvbuff, sendbuff, buff_size,
                                   cudaMemcpyDeviceToDevice,
                                   stream_vec[m_rank]));
    }

    m_client->barrier();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (root != m_rank) {
        CUDA_CHECK(cudaMemcpyAsync(recvbuff, shmadd, buff_size,
                                   cudaMemcpyHostToDevice, stream_vec[m_rank]));
    }
    m_client->barrier();
    barrier_streams_local_process();
    return MEGRAY_OK;
}

Status ShmCommunicator::broadcast(const void* sendbuff, void* recvbuff,
                                  size_t len, DType dtype, uint32_t root,
                                  std::shared_ptr<Context> ctx) {
    // only support in single machine
    if (!m_is_single_machine) {
        MEGRAY_THROW("shm all_reduce only support in single machine now");
    }
    return _shm_broadcast(sendbuff, recvbuff, len, dtype, root, ctx);
}

}  // namespace MegRay
