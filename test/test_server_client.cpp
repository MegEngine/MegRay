 /**
 * \file test/test_server_client.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "megray/core/server.h"
#include "megray/core/client.h"

TEST(TestServerClient, GetHostIP) {
    char* ip = MegRay::get_host_ip();
    ASSERT_TRUE(ip != NULL);
    ASSERT_TRUE(strlen(ip) >= 8);
}

TEST(TestServerClient, GetFreePort) {
    int port = MegRay::get_free_port();
    ASSERT_TRUE(port > 0);
}

TEST(TestServerClient, Connect) {
    const int nranks = 3;

    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    auto run = [nranks, port](int rank) {
        auto client = std::make_unique<MegRay::Client>(nranks, rank);
        auto ret = client->connect("localhost", port);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (size_t i = 0; i < nranks; i++) {
        threads[i].join();
    }
}

TEST(TestServerClient, Barrier) {
    const int nranks = 3;

    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    int counter = 0;

    auto run = [nranks, port, &counter](int rank) {
        auto client = std::make_unique<MegRay::Client>(nranks, rank);
        auto ret = client->connect("localhost", port);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        ret = client->barrier();
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        sleep(rank);
        ++counter;

        ret = client->barrier();
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        // if the barrier is not working correctly, threads that sleep
        // less seconds will arrive here earlier and counter might be
        // less than nranks
        ASSERT_EQ(nranks, counter);
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (size_t i = 0; i < nranks; i++) {
        threads[i].join();
    }
}

TEST(TestServerClient, Broadcast) {
    const int nranks = 3;
    const int root = 1;
    const int chunk_size = 10;

    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    std::string str(chunk_size * nranks, '\0');
    for (size_t i = 0; i < str.size(); i++) {
        str[i] = 'a' + i % 26;
    }
    auto expected = str.substr(root * chunk_size, chunk_size);

    auto run = [nranks, port, &str, &expected](int rank) {
        auto client = std::make_unique<MegRay::Client>(nranks, rank);
        auto ret = client->connect("localhost", port);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        const char* input = str.data() + rank * chunk_size;
        char* output = (char*)malloc(chunk_size);
        ret = client->broadcast(input, output, chunk_size, root);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        ASSERT_EQ(expected, std::string(output, chunk_size));
        free(output);
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (size_t i = 0; i < nranks; i++) {
        threads[i].join();
    }
}

TEST(TestServerClient, AllGather) {
    const int nranks = 3;
    const int chunk_size = 10;

    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    std::string str(chunk_size * nranks, '\0');
    for (size_t i = 0; i < str.size(); i++) {
        str[i] = 'a' + i % 26;
    }

    auto run = [nranks, port, &str](int rank) {
        auto client = std::make_unique<MegRay::Client>(nranks, rank);
        auto ret = client->connect("localhost", port);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        const char* input = str.data() + rank * chunk_size;
        char* output = (char*)malloc(str.size());
        ret = client->allgather(input, output, chunk_size);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        ASSERT_EQ(str, std::string(output, str.size()));
        free(output);
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (size_t i = 0; i < nranks; i++) {
        threads[i].join();
    }
}

TEST(TestServerClient, Sequence) {
    const int nranks = 3;
    const int chunk_size = 10;

    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    std::string str(chunk_size * nranks, '\0');
    for (size_t i = 0; i < str.size(); i++) {
        str[i] = 'a' + i % 26;
    }

    auto run = [nranks, port, &str](int rank) {
        auto client = std::make_unique<MegRay::Client>(nranks, rank);
        auto ret = client->connect("localhost", port);
        ASSERT_EQ(MegRay::MEGRAY_OK, ret);

        const char* input = str.data() + rank * chunk_size;
        char* output = (char*)malloc(str.size());

        // send a sequence of requets without checking output
        ASSERT_EQ(MegRay::MEGRAY_OK, client->barrier());
        ASSERT_EQ(MegRay::MEGRAY_OK, client->broadcast(input, output, chunk_size, 1));
        ASSERT_EQ(MegRay::MEGRAY_OK, client->allgather(input, output, chunk_size));
        ASSERT_EQ(MegRay::MEGRAY_OK, client->barrier());
        ASSERT_EQ(MegRay::MEGRAY_OK, client->allgather(input, output, chunk_size));
        ASSERT_EQ(MegRay::MEGRAY_OK, client->broadcast(input, output, chunk_size, 2));
        ASSERT_EQ(MegRay::MEGRAY_OK, client->allgather(input, output, chunk_size));
        ASSERT_EQ(MegRay::MEGRAY_OK, client->barrier());

        free(output);
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (size_t i = 0; i < nranks; i++) {
        threads[i].join();
    }
}
