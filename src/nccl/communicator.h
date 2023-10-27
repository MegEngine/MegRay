#pragma once

#include <memory>

#include "megray/communicator.h"

namespace MegRay {

class NcclCommunicatorPrivate;

/*!
 * a wrapper of ncclComm_t with MegRay interface
 * collective communications are performed asynchronously
 */
class NcclCommunicator : public Communicator {
public:
    NcclCommunicator(int nranks, int rank);

    ~NcclCommunicator();

    Status do_init() override;
    Status do_init(BcastCallback cb) override;
 
    Status _send(const void* sendbuff, size_t size, uint32_t rank,
                 std::shared_ptr<Context> ctx) override;

    Status _recv(void* recvbuff, size_t size, uint32_t rank,
                 std::shared_ptr<Context> ctx) override;

    Status scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                   DType dtype, uint32_t root,
                   std::shared_ptr<Context> ctx) override;

    Status gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                  DType dtype, uint32_t root,
                  std::shared_ptr<Context> ctx) override;

    Status all_to_all(const void* sendbuff, void* recvbuff, size_t len,
                      DType dtype, std::shared_ptr<Context> ctx) override;

    Status all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
                      DType dtype, std::shared_ptr<Context> ctx) override;

    Status all_reduce(const void* sendbuff, void* recvbuff, size_t len,
                      DType dtype, ReduceOp op,
                      std::shared_ptr<Context> ctx) override;

    Status reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
                          DType dtype, ReduceOp op,
                          std::shared_ptr<Context> ctx) override;

    Status broadcast(const void* sendbuff, void* recvbuff, size_t len,
                     DType dtype, uint32_t root,
                     std::shared_ptr<Context> ctx) override;

    Status reduce(const void* sendbuff, void* recvbuff, size_t len, DType dtype,
                  ReduceOp op, uint32_t root,
                  std::shared_ptr<Context> ctx) override;
    
    Status group_start() override;
    Status group_end() override;
    
private:
    std::unique_ptr<NcclCommunicatorPrivate> m_nccl;
};

}  // namespace MegRay
