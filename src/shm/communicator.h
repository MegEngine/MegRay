#pragma once
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "megray/common.h"
#include "megray/communicator.h"
#include "shm_ipc.h"

#ifdef MEGRAY_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif
namespace MegRay {

struct Op {
    void* shm_buffer;
    volatile int *send_signal, *reduce_signal, *shm_mutex, *op_begin;
    const void* sendbuff;
    void* recvbuff;
    size_t len, k;
    DType dtype;
    ReduceOp op;
    bool is_nop = false;
};

template <typename T>
class BlockingQueue {
public:
    T pop();
    void push(T op);

private:
    std::queue<T> m_ops;
    std::mutex m_queue_mtx;
    std::condition_variable m_queue_cv;
};

template <typename T>
T BlockingQueue<T>::pop() {
    std::unique_lock<std::mutex> lock{m_queue_mtx};
    m_queue_cv.wait(lock, [this]() { return m_ops.size() > 0; });
    T ret = m_ops.front();
    m_ops.pop();
    return ret;
}

template <typename T>
void BlockingQueue<T>::push(T op) {
    std::unique_lock<std::mutex> lock{m_queue_mtx};
    m_ops.push(op);
    lock.unlock();
    m_queue_cv.notify_one();
}

struct Shm {
    void* addr;
    key_t key;
    size_t size;
    int shmid;
};

/*!
 * simple implementation of collective communications using shared memory
 * a shared memory communicator corresponds to a tcp communicator
 * */

class ShmCommunicator : public Communicator {
public:
    // constructor
    ShmCommunicator(int nranks, int rank);
    // deconstructor
    ~ShmCommunicator();
    // init function
    Status do_init() override;
    Status do_init(BcastCallback) override;
    // _send function
    Status _send(const void* sendbuff, size_t size, uint32_t rank,
                 std::shared_ptr<Context> ctx) override;
    // _recv function
    Status _recv(void* recvbuff, size_t size, uint32_t rank,
                 std::shared_ptr<Context> ctx) override;

    // the scatter operation
    Status scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                   DType dtype, uint32_t root,
                   std::shared_ptr<Context> ctx) override;
    // the gather option
    Status gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                  DType dtype, uint32_t root,
                  std::shared_ptr<Context> ctx) override;
    // the all_to_all operation
    Status all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                      DType dtype, std::shared_ptr<Context> ctx) override;
    // the all_gather operation
    Status all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                      DType dtype, std::shared_ptr<Context> ctx) override;
    // the all_reduce operation
    Status all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                      DType dtype, ReduceOp op,
                      std::shared_ptr<Context> ctx) override;
    // the reduce scatter operation
    Status reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                          DType dtype, ReduceOp op,
                          std::shared_ptr<Context> ctx) override;
    // the broadcast operation
    Status broadcast(const void* sendbuff, void* recvbuff, size_t len,
                     DType dtype, uint32_t root,
                     std::shared_ptr<Context> ctx) override;
    // the reduce operation
    Status reduce(const void* sendbuff, void* recvbuff, size_t len, DType dtype,
                  ReduceOp op, uint32_t root,
                  std::shared_ptr<Context> ctx) override;

    Status group_start() override;
    Status group_end() override;

protected:
    // allocate cuda stream for parllel
    void alloc_cuda_stream();
    // destroy parllel stream
    void free_cuda_stream();
    // barrier current streams
    void barrier_streams_local_process();
    // use to free shared memory
    void free_shm();
    // alloc shared memory with given size
    void* alloc_shm(size_t size);

    // using shm broadcast data
    Status _shm_broadcast(const void* sendbuff, void* recvbuff, size_t len,
                          DType dtype, uint32_t root,
                          std::shared_ptr<Context> ctx);
    // using shm all reduce
    Status _shm_all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                           DType dtype, ReduceOp op,
                           std::shared_ptr<Context> ctx);
    // using shm all gather algorithm
    Status _shm_all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                           DType dtype, std::shared_ptr<Context> ctx);
    // using shm all to all algorithm
    Status _shm_all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                           DType dtype, std::shared_ptr<Context> ctx);
    // using shm gather algorithm
    Status _shm_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                       DType dtype, uint32_t root,
                       std::shared_ptr<Context> ctx);
    // using shm reduce algorithm
    Status _shm_reduce(const void* sendbuff, void* recvbuff, size_t len,
                       DType dtype, ReduceOp op, uint32_t root,
                       std::shared_ptr<Context> ctx);
    // using shm to do reduce scatter
    Status _shm_reduce_scatter(const void* sendbuff, void* recvbuff,
                               size_t recvlen, DType dtype, ReduceOp op,
                               std::shared_ptr<Context> ctx);
    // using shm to recv data
    Status _irecv(void* recvbuff, size_t len, uint32_t rank,
                  std::shared_ptr<Context> ctx);
    // using shm to send data
    Status _isend(const void* sendbuff, size_t len, uint32_t rank,
                  std::shared_ptr<Context> ctx);
    // used for barrier the sum
    void _shm_barrier_sum(volatile int* mutex);
    // used for barrier the all reduce
    void _shm_barrier(volatile int* mutex);

    void persistent_thread();

private:
    // parllel mutex
    std::mutex m_requests_mtx;
    // use for parllell in device
    std::vector<cudaStream_t> stream_vec;
    std::vector<Shm> m_shm_list;
    // single flag that label in single machine
    bool m_is_single_machine = true;

    BlockingQueue<Op> m_oplist;
    std::thread m_persistent_thread;
};

}  // namespace MegRay
