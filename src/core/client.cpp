/**
 * \file src/client.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megray/core/client.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>

namespace MegRay {

Client::Client(uint32_t nranks, uint32_t rank) :
        m_nranks(nranks), m_rank(rank), m_connected(false) {
}

Client::~Client() {
}

Status Client::connect(const char* master_ip, int port) {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_connected) {
        MEGRAY_ERROR("Client already connected");
        return MEGRAY_INVALID_USAGE;
    }

    // create socket
    SYS_CHECK_RET(socket(AF_INET, SOCK_STREAM, 0), -1, m_conn);

    // set server_addr
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    SYS_CHECK(inet_pton(AF_INET, master_ip, &server_addr.sin_addr), -1);

    // connect
    SYS_CHECK(::connect(m_conn, (struct sockaddr*)&server_addr, sizeof(server_addr)), -1);

    // send client rank
    SYS_CHECK(send(m_conn, &m_rank, sizeof(uint32_t), 0), -1);

    // recv ack from server
    uint32_t ack;
    SYS_CHECK(recv(m_conn, &ack, sizeof(uint32_t), MSG_WAITALL), -1);

    m_connected = true;
    return MEGRAY_OK;
}

Status Client::barrier() {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (!m_connected) {
        MEGRAY_ERROR("Client not connected");
        return MEGRAY_INVALID_USAGE;
    }

    // send request_id
    uint32_t request_id = 1;
    SYS_CHECK(send(m_conn, &request_id, sizeof(uint32_t), 0), -1);

    // recv ack
    uint32_t ack;
    SYS_CHECK(recv(m_conn, &ack, sizeof(uint32_t), MSG_WAITALL), -1);

    return MEGRAY_OK;
}

Status Client::broadcast(const void* sendbuff, void* recvbuff, size_t len, uint32_t root) {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (!m_connected) {
        MEGRAY_ERROR("Client not connected");
        return MEGRAY_INVALID_USAGE;
    }

    // send request_id
    uint32_t request_id = 2;
    SYS_CHECK(send(m_conn, &request_id, sizeof(uint32_t), 0), -1);

    // send root
    SYS_CHECK(send(m_conn, &root, sizeof(uint32_t), 0), -1);

    // send len
    uint64_t len64 = len;
    SYS_CHECK(send(m_conn, &len64, sizeof(uint64_t), 0), -1);

    // send data
    if (m_rank == root) {
        SYS_CHECK(send(m_conn, sendbuff, len, 0), -1);
    }

    // recv data
    SYS_CHECK(recv(m_conn, recvbuff, len, MSG_WAITALL), -1);

    return MEGRAY_OK;
}

Status Client::allgather(const void* sendbuff, void* recvbuff, size_t sendlen) {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (!m_connected) {
        MEGRAY_ERROR("Client not connected");
        return MEGRAY_INVALID_USAGE;
    }

    // send request_id
    uint32_t request_id = 3;
    SYS_CHECK(send(m_conn, &request_id, sizeof(uint32_t), 0), -1);

    // send sendlen
    uint64_t sendlen64 = sendlen;
    SYS_CHECK(send(m_conn, &sendlen64, sizeof(uint64_t), 0), -1);

    // send data
    SYS_CHECK(send(m_conn, sendbuff, sendlen, 0), -1);

    // recv data
    SYS_CHECK(recv(m_conn, recvbuff, sendlen * m_nranks, MSG_WAITALL), -1);

    return MEGRAY_OK;
}

} // namespace MegRay
