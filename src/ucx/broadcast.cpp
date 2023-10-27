#include "communicator.h"

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status UcxCommunicator::broadcast(const void* sendbuff, void* recvbuff,
                                  size_t len, DType dtype, uint32_t root,
                                  std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t size = get_dtype_size(dtype);
    if (m_rank == root) {
        CUDA_CHECK(cudaMemcpy(recvbuff, sendbuff, len * size,
                              cudaMemcpyDeviceToDevice));
    }

    // offset to make sure virtual_root is 0
    auto virtual_rank = ring_sub(m_rank, root, m_nranks);

    // we need d rounds to broadcast data
    size_t d = 0, t = m_nranks - 1;
    while (t > 0) {
        ++d;
        t >>= 1;
    }

    // begin with one node with msg
    // on each round every node with msg sends to one node without msg
    // on the 1-st round , node B0000000 sends msg to node B1000000
    // on the i-th round , node Bxxx0000 sends msg to node Bxxx1000
    // on the last round , node Bxxxxxx0 sends msg to node Bxxxxxx1
    int mask = (1 << d) - 1;
    for (int i = d - 1; i >= 0; --i) {
        int bit = 1 << i;
        mask = mask ^ bit;
        if ((virtual_rank & mask) == 0) {
            if ((virtual_rank & bit) == 0) {
                auto virtual_dest = virtual_rank ^ bit;
                auto actual_dest = ring_add(virtual_dest, root, m_nranks);
                if (virtual_dest < m_nranks) {  // valid dest
                    MEGRAY_CHECK(_isend(recvbuff, len * size, actual_dest));
                    MEGRAY_CHECK(_flush());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            } else {
                auto virtual_src = virtual_rank ^ bit;
                auto actual_src = ring_add(virtual_src, root, m_nranks);
                if (virtual_src < m_nranks) {  // valid src
                    MEGRAY_CHECK(_irecv(recvbuff, len * size, actual_src));
                    MEGRAY_CHECK(_flush());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
        }
    }
    return MEGRAY_OK;
}

}  // namespace MegRay
