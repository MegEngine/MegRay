/**
 * \file test/test_shm.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <fcntl.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstring>
#include <string>
#include <thread>
#include "megray.h"
#include "megray/common.h"

#ifdef MEGRAY_WITH_SHM

#include "../src/shm/shm_ipc.h"

TEST(TestShm, write) {
    std::string test = "hello world!";
    void* result_ptr = malloc(test.size());
    MegRay::ShmMemory mem(MegRay::SEND_RECV(0, 1, 0), test.size(), true);
    auto send_cb = [=](MegRay::raw_void_ptr shm_ptr) {
        memcpy(shm_ptr, test.c_str(), test.size());
    };
    mem.shm_rw_startegy(send_cb);
    MegRay::ShmMemory recv_mem(MegRay::SEND_RECV(0, 1, 0), test.size(), true);
    auto recv_cb = [=](MegRay::raw_void_ptr shm_ptr) {
        memcpy(result_ptr, shm_ptr, test.size());
    };
    mem.shm_rw_startegy(recv_cb);
    free(result_ptr);
}

TEST(TestShm, send_recv) {
    std::string name = "hello world!";
    void* result_ptr = malloc(name.size());
    auto send_cb = [=](MegRay::raw_void_ptr shm_ptr) {
        memcpy(shm_ptr, name.c_str(), name.size());
    };
    auto recv_cb = [=](MegRay::raw_void_ptr shm_ptr) {
        memcpy(result_ptr, shm_ptr, name.size());
    };
    MegRay::RingSender send;
    MegRay::RingRecver recv;
    ASSERT_EQ(MegRay::MEGRAY_OK, send.Notify(0, 1, 0, name.size(), send_cb));
    ASSERT_EQ(MegRay::MEGRAY_OK, recv.Update(0, 1, 0, name.size(), recv_cb));
}

TEST(TestShm, multi_thread_send_recv) {
    std::string name = "hello world!";
    void* result_ptr = malloc(name.size());

    auto send_thread = [=]() {
        auto send_cb = [=](MegRay::raw_void_ptr shm_ptr) {
            memcpy(shm_ptr, name.c_str(), name.size());
        };
        MegRay::RingSender send;
        ASSERT_EQ(MegRay::MEGRAY_OK,
                  send.Notify(0, 1, 0, name.size(), send_cb));
    };
    auto recv_thread = [=]() {
        auto recv_cb = [=](MegRay::raw_void_ptr shm_ptr) {
            memcpy(result_ptr, shm_ptr, name.size());
        };
        MegRay::RingRecver recv;
        ASSERT_EQ(MegRay::MEGRAY_OK,
                  recv.Update(0, 1, 0, name.size(), recv_cb));
    };
    std::vector<std::thread> threads;
    threads.push_back(std::thread(recv_thread));
    threads.push_back(std::thread(send_thread));
    for (auto& th : threads) {
        th.join();
    }
}

#endif