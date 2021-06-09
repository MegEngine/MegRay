/**
 * \file src/shm/shm_ipc.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "megray/common.h"
#include "megray/cuda_context.h"

namespace MegRay {
std::string SEND_RECV(int from, int to, int channel);
#define MEGRAY_VERSION "MEGRAY VERSION 0.0.1"
using raw_void_ptr = void*;
using shm_wr_startegy_cb = std::function<void(raw_void_ptr)>;
/*
 *  a protocal used to block memory access
 * */
class ProtocalHead {
public:
    // the send done flag and will notify any
    // host listen to this message
    bool send_finish{false};
    // when write done set the flag to true
    bool recv_finish{false};
    // the protocal version header which is to
    // make sure that the memory is initialized
    char version[22] = MEGRAY_VERSION;
    // an static check function
    // which is used to check whether the protocal
    // is initialize
    static bool ProtocalValid(volatile ProtocalHead* head) {
        return strncasecmp((char*)head->version, MEGRAY_VERSION,
                           strlen(MEGRAY_VERSION)) == 0;
    }
};

/*
 * an abstruct of shared memory class
 * which can be call by send and recver
 *
 * */
class ShmMemory {
public:
    // init the shared memory
    ShmMemory(std::string&& str, size_t size, bool flag);
    // destory the shared memory
    ~ShmMemory() noexcept(false);
    // an abstruct interface for read and write
    void shm_rw_startegy(shm_wr_startegy_cb func) {
        func((uint8_t*)ptr + sizeof(ProtocalHead));
    }

    // allocate the memory
    int allocate();

    // called by recv process
    // which is followed by the idea of mpi
    // the recv process will be block and the send process will not block
    void wait_send_done_block();

    // block to wait the current recv done
    Status wait_current_channel_nobusy_block();

    // to notify that the publish process is done
    void notify_one();

    // set the flag to ture
    void recv_finish();

private:
    // the shared memory name
    std::string name;
    // the shared memory size
    size_t buf_size;
    // the shm ptr
    raw_void_ptr ptr;
    // the shared memory file descripter
    int fd{-1};
    bool is_pub;
};

/*
 *  an abstruct for the sender
 * */
class Sender {
public:
    virtual ~Sender() {}
    virtual Status Notify(int from, int to, int channel, size_t len,
                          shm_wr_startegy_cb func) = 0;

protected:
    std::unique_ptr<ShmMemory> _mem{nullptr};
};

/*
 *
 *  the ring sender class which will work in multiprocess
 *
 * */
class RingSender : public Sender {
public:
    // notify the recv
    virtual Status Notify(int from, int to, int channel, size_t len,
                          shm_wr_startegy_cb func) override {
        MEGRAY_CHECK(_send(from, to, channel, len, func));
        _mem->notify_one();
        return MEGRAY_OK;
    }
    // send the buffer to shm
    Status _send(int from, int to, int channel, size_t len,
                 shm_wr_startegy_cb func) {
        _mem = std::make_unique<ShmMemory>(SEND_RECV(from, to, channel), len,
                                           true);
        _mem->shm_rw_startegy(func);
        return MEGRAY_OK;
    }
};

/*
 *   an abstruct for the recver
 *
 * */
class Recver {
public:
    virtual ~Recver() {}
    virtual Status Update(int from, int to, int channel, size_t len,
                          shm_wr_startegy_cb func) = 0;

protected:
    std::unique_ptr<ShmMemory> _mem{nullptr};
};

/*
 *  the ring recv process
 *  which will block when the send process do
 *
 * */
class RingRecver : public Recver {
public:
    // the recv thread
    // will block to get the result
    // TODO: when update done close the channel, but this will lead to
    // a lot of overhead

    virtual Status Update(int from, int to, int channel, size_t len,
                          shm_wr_startegy_cb func) override {
        MEGRAY_CHECK(_recv(from, to, channel, len, func));
        _mem->recv_finish();
        return MEGRAY_OK;
    }

    // recv from device stream
    Status _recv(int from, int to, int channel, size_t len,
                 shm_wr_startegy_cb func) {
        _mem = std::make_unique<ShmMemory>(SEND_RECV(from, to, channel), len,
                                           false);
        // this will block the thread
        _mem->wait_send_done_block();
        _mem->shm_rw_startegy(func);
        return MEGRAY_OK;
    }
};

class ChannelManager {
public:
    int get_channel(int from, int to) {
        std::string temp_name = std::to_string(from) + "_" + std::to_string(to);
        auto it = channel_map.find(temp_name);
        if (it != channel_map.end()) {
            return ++(it->second);
        } else {
            channel_map.emplace(temp_name, 0);
            return 0;
        }
    }

private:
    std::unordered_map<std::string, int> channel_map;
};

}  // namespace MegRay
