/**
 * \file test/test_base.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/megray.h"

template <typename T>
void run_test(int nranks, MegRay::Backend backend,
              std::vector<std::vector<T>>& inputs,
              std::vector<std::vector<T>>& expect_outputs,
              std::function<void(std::shared_ptr<MegRay::Communicator>, int, int,
                                 std::vector<T>&, std::vector<T>&)>
                      main_func) {
    std::vector<std::shared_ptr<MegRay::Communicator>> comms(nranks);
    std::vector<std::vector<T>> outputs(nranks);

    int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    for (int i = 0; i < nranks; i++) {
        comms[i] = MegRay::get_communicator(nranks, i, backend);
        outputs[i].resize(expect_outputs[i].size());
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < nranks; i++) {
        threads.push_back(std::thread(main_func, comms[i], port, i,
                                      std::ref(inputs[i]),
                                      std::ref(outputs[i])));
    }

    for (int i = 0; i < nranks; i++) {
        threads[i].join();
    }

    for (int i = 0; i < nranks; i++) {
        for (size_t j = 0; j < expect_outputs[i].size(); j++) {
            ASSERT_FLOAT_EQ(expect_outputs[i][j], outputs[i][j]);
        }
    }
}
