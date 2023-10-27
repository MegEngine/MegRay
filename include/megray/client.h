#pragma once

#include <mutex>

#include "megray/common.h"

namespace MegRay {

/*!
 * synchronize meta information with megray server
 */
class Client {
public:
    Client(uint32_t nranks, uint32_t rank);

    ~Client();

    Status connect(const char* master_ip, int port);

    // block until all ranks reach this barrier
    Status barrier();

    // the length of sendbuff = the length of recvbuff = len
    Status broadcast(const void* sendbuff, void* recvbuff, size_t sendlen,
                     uint32_t root);

    // the length of sendbuff = sendlen
    // the length of recvbuff = sendlen * m_nranks
    Status allgather(const void* sendbuff, void* recvbuff, size_t sendlen);

private:
    uint32_t m_nranks;
    uint32_t m_rank;
    bool m_connected = false;
    int m_conn;
    std::mutex m_mutex;
};

}  // namespace MegRay
