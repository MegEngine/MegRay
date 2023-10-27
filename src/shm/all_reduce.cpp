#include "communicator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "megray/common.h"
#include "megray/debug.h"
#include "megray/server.h"
#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status ShmCommunicator::_shm_all_reduce(const void* sendbuff, void* recvbuff,
                                        size_t len, DType dtype, ReduceOp op,
                                        std::shared_ptr<Context> ctx) {
    // prepare
    size_t k = 2;
    size_t chunks = m_nranks * k;
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    size_t size = get_dtype_size(dtype);
    size_t buff_size = len * size;

    void* shmadd = alloc_shm(buff_size * 2 + m_nranks * sizeof(int) * 4);

    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();

    // compute chunk sizes
    size_t quotient = len / chunks;
    size_t remainder = len % chunks;
    std::vector<size_t> chunk_sizes(chunks, quotient);
    for (size_t i = 0; i < remainder; i++) {
        chunk_sizes[i]++;
    }

    // compute chunk offsets
    std::vector<size_t> offsets(chunks, 0);
    for (size_t i = 1; i < chunks; i++) {
        offsets[i] = offsets[i - 1] + chunk_sizes[i - 1] * size;
    }
    size_t tmp_size = 0;
    for (size_t i = 0; i < k; i++) {
        tmp_size += chunk_sizes[m_rank * k + i];
    }

    Op all_reduce_op;
    all_reduce_op.dtype = dtype;
    all_reduce_op.k = k;
    all_reduce_op.len = len;
    all_reduce_op.op = op;
    all_reduce_op.recvbuff = recvbuff;
    all_reduce_op.sendbuff = sendbuff;
    all_reduce_op.op_begin = (volatile int*)shmadd+m_rank;
    all_reduce_op.shm_mutex = (int*)shmadd + m_nranks;
    all_reduce_op.send_signal = (int*)shmadd + m_nranks*2;
    all_reduce_op.reduce_signal = (int*)shmadd + m_nranks*3;
    all_reduce_op.shm_buffer = (int*)shmadd + m_nranks*4;
    m_oplist.push(all_reduce_op);

    // launch kernel

    // wait begin
    stream_wait_signal(all_reduce_op.op_begin, 1, stream);
    stream_set_signal(all_reduce_op.op_begin, 0, stream);
    CUDA_ASSERT(cudaMemcpyAsync((char*)all_reduce_op.shm_buffer + offsets[m_rank * k],
                                (char*)sendbuff + offsets[m_rank * k],
                                tmp_size * size, cudaMemcpyDeviceToHost,
                                stream));
    stream_set_signal(all_reduce_op.send_signal + m_rank, k, stream);
    stream_set_signal(all_reduce_op.reduce_signal + m_rank, k, stream);
    uint32_t work_rank;
    work_rank = ring_add(m_rank * k, k, chunks);
    size_t right_rank = ring_add(m_rank, 1, m_nranks);
    // wait last reduction done, copy ith part
    for (size_t i = k+1; i <= chunks; i++) {
        stream_wait_signal(all_reduce_op.reduce_signal + right_rank, i-k, stream);
        CUDA_ASSERT(cudaMemcpyAsync(
                (char*)all_reduce_op.shm_buffer + buff_size + offsets[work_rank],
                (char*)sendbuff + offsets[work_rank],
                chunk_sizes[work_rank] * size, cudaMemcpyDeviceToHost,
                stream));
        stream_set_signal(all_reduce_op.send_signal + m_rank, i, stream);
        work_rank = ring_add(work_rank, 1, chunks);
    }
    for (int i = 0;i < k;i++) {
        work_rank = ring_sub(m_rank*k+i, k, chunks);
        int now_rank = m_rank;
        for (int j = 0;j < m_nranks;j++) {
            stream_wait_signal(all_reduce_op.reduce_signal + now_rank, chunks-k+i+1, stream);
            CUDA_ASSERT(cudaMemcpyAsync(
                (char*)recvbuff + offsets[work_rank],
                (char*)all_reduce_op.shm_buffer + offsets[work_rank],
                chunk_sizes[work_rank] * size, cudaMemcpyHostToDevice,
                stream));
            work_rank = ring_add(work_rank, k, chunks);
            now_rank = ring_add(now_rank, 1, m_nranks);
        }
    }
    stream_set_signal(all_reduce_op.send_signal + m_rank, chunks+1, stream);

    return MEGRAY_OK;
}

Status ShmCommunicator::all_reduce(const void* sendbuff, void* recvbuff,
                                   size_t len, DType dtype, ReduceOp op,
                                   std::shared_ptr<Context> ctx) {
    // only support in single machine
    if (!m_is_single_machine) {
        MEGRAY_THROW("shm all_reduce only support in single machine now");
    }
    return _shm_all_reduce(sendbuff, recvbuff, len, dtype, op, ctx);
}

}  // namespace MegRay
