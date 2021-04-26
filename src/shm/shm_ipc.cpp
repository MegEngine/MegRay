/**
 * \file src/shm/shm_ipc.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "shm_ipc.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <functional>
#include <mutex>
#include <string>

namespace MegRay {

std::string SEND_RECV(int from, int to, int channel) {
    return "megray_send_recv_" + std::to_string(from) + "_" +
           std::to_string(to) + "_" + std::to_string(channel);
}

/******************** ShmMemory start *******************/

ShmMemory::ShmMemory(std::string&& str, size_t size, bool flag)
        : name(str),
          buf_size(sizeof(ProtocalHead) + size),
          is_pub(flag),
          ptr(MAP_FAILED) {
    // allocate shm resources
    if (is_pub) {
        // spain lock to make sure there is one channel work
        // will block the send thread now
        // MEGRAY_CHECK(wait_current_channel_nobusy_block());
        SYS_ASSERT_RET(
                shm_open(name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), -1,
                fd);
    } else {
        // use spine lock to break the recv thread
        while (true) {
            fd = shm_open(name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
            if (fd >= 0) {
                break;
            }
        }
    }
    // make sure all send and recv can get the same size
    // of mmap
    SYS_ASSERT(allocate(), -1);
    SYS_ASSERT_RET(mmap(NULL /* let kernal map the address*/, buf_size,
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0),
                   MAP_FAILED, ptr);
    close(fd);
    fd = -1;
    // reset the header in setup process
    if (is_pub) {
        ProtocalHead head;
        memcpy(ptr, &head, sizeof(head));
    }
    // register ptr to cuda
    CUDA_ASSERT(cudaHostRegister(ptr, buf_size, cudaHostRegisterMapped));
}

ShmMemory::~ShmMemory() noexcept(false) {
    CUDA_ASSERT(cudaHostUnregister(ptr));
    SYS_ASSERT(munmap(ptr, buf_size), -1);
    if (is_pub) {
        // publish thread doesn't need to unlink
    } else {
        // unlink the shared memory resources
        if (name.size())
            SYS_ASSERT(shm_unlink(name.c_str()), -1);
    }
}

int ShmMemory::allocate() {
    int err = posix_fallocate(fd, 0, buf_size);
    if (err) {
        errno = err;
        return -1;
    }
    return 0;
}

void ShmMemory::wait_send_done_block() {
    while (true) {
        ProtocalHead* head = (ProtocalHead*)ptr;
        if (ProtocalHead::ProtocalValid(head)) {  // the protocal is done
            if (head->send_finish) {
                return;
            }
        }
    }
}

Status ShmMemory::wait_current_channel_nobusy_block() {
    int flag = 0;
    while (true) {
        fd = shm_open(name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    return MEGRAY_OK;
}

void ShmMemory::notify_one() {
    ProtocalHead* head = (ProtocalHead*)ptr;
    head->send_finish = true;
}

void ShmMemory::recv_finish() {
    ProtocalHead* head = (ProtocalHead*)ptr;
    head->send_finish = false;
    head->recv_finish = true;
}

}  // namespace MegRay
