/**
 * \file src/server.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megray/core/server.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <thread>

namespace MegRay {

/************************ get_host_ip ************************/

char* get_host_ip() {
    const char* device = getenv("MEGRAY_NET_DEVICE");
    if (device and strcmp(device, "lo") == 0) {
        MEGRAY_ERROR("illegal net device: lo\n");
        MEGRAY_THROW("invalid argument");
    }

    struct ifaddrs *ifa;
    SYS_ASSERT(getifaddrs(&ifa), -1);

    for (struct ifaddrs* p = ifa; p != NULL; p = p->ifa_next) {
        if (p->ifa_addr and p->ifa_addr->sa_family == AF_INET and p->ifa_name) {
            const char* name = p->ifa_name;
            if (strcmp(name, "lo") != 0 and
                    (device == NULL or strcmp(name, device) == 0)) {
                struct sockaddr_in* sin = (struct sockaddr_in*)p->ifa_addr;
                const char* host_ip = inet_ntoa(sin->sin_addr);
                MEGRAY_INFO("using net device %s (%s)", name, host_ip);
                char* ret = new char[strlen(host_ip) + 1];
                strcpy(ret, host_ip);
                freeifaddrs(ifa);
                return ret;
            }
        }
    }

    if (device) {
        MEGRAY_ERROR("failed to get host ip for device %s", device);
    } else {
        MEGRAY_ERROR("failed to get host ip");
    }
    MEGRAY_THROW("system error");
    return nullptr;
}

/************************ get_free_port ************************/

int get_free_port() {
    // create socket
    int sock;
    SYS_ASSERT_RET(socket(AF_INET, SOCK_STREAM, 0), -1, sock);

    // set address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(0);

    // bind
    SYS_ASSERT(bind(sock, (struct sockaddr*)&addr, sizeof(addr)), -1);

    // get port
    socklen_t len = sizeof(addr);
    SYS_ASSERT(getsockname(sock, (struct sockaddr*)&addr, &len), -1);
    int port = ntohs(addr.sin_port);

    // close
    SYS_ASSERT(close(sock), -1);

    return port;
}

/************************ create_server ************************/

void serve_barrier(uint32_t nranks, int* conns);

void serve_broadcast(uint32_t nranks, int* conns);

void serve_allgather(uint32_t nranks, int* conns);

void server_thread(int listenfd, uint32_t nranks) {
    int conns[nranks];

    for (uint32_t i = 0; i < nranks; i++) {
        // establish connection
        int conn;
        SYS_ASSERT_RET(accept(listenfd, (struct sockaddr*)NULL, NULL), -1, conn);

        // recv rank and save into conns
        uint32_t rank;
        SYS_ASSERT(recv(conn, &rank, sizeof(uint32_t), MSG_WAITALL), -1);
        conns[rank] = conn;
    }

    // send ack to clients
    uint32_t ack = 0;
    for (uint32_t i = 0; i < nranks; i++) {
        SYS_ASSERT(send(conns[i], &ack, sizeof(uint32_t), 0), -1);
    }

    while (true) {
        // receive a request from rank 0
        uint32_t request_id;
        SYS_ASSERT(recv(conns[0], &request_id, sizeof(uint32_t), MSG_WAITALL), -1);

        if (request_id == 1) {
            serve_barrier(nranks, conns);
        } else if (request_id == 2) {
            serve_broadcast(nranks, conns);
        } else if (request_id == 3) {
            serve_allgather(nranks, conns);
        } else {
            MEGRAY_ERROR("unexpected request id: %d", request_id);
            MEGRAY_THROW("unexpected error");
        }
    }
}

Status create_server(uint32_t nranks, int port) {
    // create socket
    int listenfd;
    SYS_CHECK_RET(socket(AF_INET, SOCK_STREAM, 0), -1, listenfd);

    // set server_addr
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(port);

    // bind and listen
    int opt = 1;
    SYS_CHECK(setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)), -1);
    SYS_CHECK(bind(listenfd, (struct sockaddr*)&server_addr, sizeof(server_addr)), -1);
    SYS_CHECK(listen(listenfd, nranks), -1);

    // start server thread
    std::thread th(server_thread, listenfd, nranks);
    th.detach();

    return MEGRAY_OK;
}

/************************ barrier ************************/

void serve_barrier(uint32_t nranks, int* conns) {
    uint32_t request_id;

    // recv other requests
    for (uint32_t rank = 1; rank < nranks; rank++) {
        SYS_ASSERT(recv(conns[rank], &request_id, sizeof(uint32_t), MSG_WAITALL), -1);
        MEGRAY_ASSERT(request_id == 1, "inconsistent request_id from rank %d", rank);
    }

    // send ack
    uint32_t ack = 0;
    for (uint32_t rank = 0; rank < nranks; rank++) {
        SYS_ASSERT(send(conns[rank], &ack, sizeof(uint32_t), 0), -1);
    }
}

/************************ broadcast ************************/

void serve_broadcast(uint32_t nranks, int* conns) {
    uint32_t request_id, root, root0;
    uint64_t len, len0;

    // recv request 0
    SYS_ASSERT(recv(conns[0], &root0, sizeof(uint32_t), MSG_WAITALL), -1);
    SYS_ASSERT(recv(conns[0], &len0, sizeof(uint64_t), MSG_WAITALL), -1);

    // recv other requests
    for (uint32_t rank = 1; rank < nranks; rank++) {
        SYS_ASSERT(recv(conns[rank], &request_id, sizeof(uint32_t), MSG_WAITALL), -1);
        MEGRAY_ASSERT(request_id == 2, "inconsistent request_id from rank %d", rank);

        SYS_ASSERT(recv(conns[rank], &root, sizeof(uint32_t), MSG_WAITALL), -1);
        MEGRAY_ASSERT(root == root0, "inconsistent root from rank %d", rank);

        SYS_ASSERT(recv(conns[rank], &len, sizeof(uint64_t), MSG_WAITALL), -1);
        MEGRAY_ASSERT(len == len0, "inconsistent len from rank %d", rank);
    }

    // recv data from root
    void* data = malloc(len);
    SYS_ASSERT(recv(conns[root], data, len, MSG_WAITALL), -1);

    // send data to clients
    for (uint32_t rank = 0; rank < nranks; rank++) {
        SYS_ASSERT(send(conns[rank], data, len, 0), -1);
    }

    free(data);
}

/************************ allgather ************************/

void serve_allgather(uint32_t nranks, int* conns) {
    uint32_t request_id;
    uint64_t len, len0;

    // recv request 0
    SYS_ASSERT(recv(conns[0], &len0, sizeof(uint64_t), MSG_WAITALL), -1);

    // recv other requests
    for (uint32_t rank = 1; rank < nranks; rank++) {
        SYS_ASSERT(recv(conns[rank], &request_id, sizeof(uint32_t), MSG_WAITALL), -1);
        MEGRAY_ASSERT(request_id == 3, "inconsistent request_id from rank %d", rank);

        SYS_ASSERT(recv(conns[rank], &len, sizeof(uint64_t), MSG_WAITALL), -1);
        MEGRAY_ASSERT(len == len0, "inconsistent len from rank %d", rank);
    }

    // recv data
    void* data = malloc(len * nranks);
    for (uint32_t rank = 0; rank < nranks; rank++) {
        char* ptr = (char*)data + rank * len;
        SYS_ASSERT(recv(conns[rank], ptr, len, MSG_WAITALL), -1);
    }

    // send data to clients
    for (uint32_t rank = 0; rank < nranks; rank++) {
        SYS_ASSERT(send(conns[rank], data, len * nranks, 0), -1);
    }

    free(data);
}

} // namespace MegRay
