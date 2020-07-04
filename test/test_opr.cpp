/**
 * \file test/test_opr.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "test_base.h"

TEST(TestNcclCommunicator, Init) {
    const int nranks = 3;
    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    auto run = [&](int rank) {
        cudaSetDevice(rank);
        auto comm = MegRay::get_communicator(nranks, rank, MegRay::MEGRAY_NCCL);
        ASSERT_EQ(MegRay::MEGRAY_OK, comm->init("localhost", port));
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (size_t i = 0; i < nranks; i++) {
        threads[i].join();
    }
}

TEST(TestUcxCommunicator, Init) {
    const int nranks = 3;
    const int port = MegRay::get_free_port();
    auto ret = MegRay::create_server(nranks, port);
    ASSERT_EQ(MegRay::MEGRAY_OK, ret);

    auto run = [&](int rank) {
        cudaSetDevice(rank);
        auto comm = MegRay::get_communicator(nranks, rank, MegRay::MEGRAY_UCX);
        ASSERT_EQ(MegRay::MEGRAY_OK, comm->init("localhost", port));
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < nranks; i++) {
        threads.push_back(std::thread(run, i));
    }

    for (int i = 0; i < nranks; i++) {
        threads[i].join();
    }
}

TEST(TestOpr, SendRecv) {
    std::string msg("test_message");
    const int nranks = 2;
    const size_t len = msg.size();

    std::vector<std::vector<char>> inputs(nranks);
    std::vector<std::vector<char>> expected_outputs(nranks);

    for (size_t i = 0; i < len; i++) {
        inputs[0].push_back(msg[i]);
        expected_outputs[1].push_back(msg[i]);
    }

    auto run = [len](std::shared_ptr<MegRay::Communicator> comm,
                     int port, int rank,
                     std::vector<char>& input,
                     std::vector<char>& output) -> void {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init("localhost", port);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void* ptr;
        CUDA_ASSERT(cudaMalloc(&ptr, len));

        if (rank == 0) {  // send
            CUDA_ASSERT(cudaMemcpy(ptr, input.data(), len, cudaMemcpyHostToDevice));
            comm->send(ptr, len, 1, ctx);
            CUDA_ASSERT(cudaStreamSynchronize(stream));
        } else {  // recv
            comm->recv(ptr, len, 0, ctx);
            CUDA_ASSERT(cudaStreamSynchronize(stream));
            CUDA_ASSERT(cudaMemcpy(output.data(), ptr, len, cudaMemcpyDeviceToHost));
        }
    };

    run_test<char>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs, run);
    run_test<char>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs, run);
}

TEST(TestOpr, Scatter) {
    const int nranks = 3;
    const size_t recvlen = 10;
    const int root = 1;

    std::vector<std::vector<float>> inputs(nranks);
    std::vector<std::vector<float>> outputs(nranks);
    for (size_t i = 0; i < nranks; i++) {
        for (size_t j = 0; j < recvlen; j++) {
            float val = 1.0 * (i + 1) * (j + 2);
            inputs[root].push_back(val);
            outputs[i].push_back(val);
        }
    }

    auto run = [nranks, recvlen, root](std::shared_ptr<MegRay::Communicator> comm,
                                       int port, int rank,
                                       std::vector<float>& input,
                                       std::vector<float>& output) -> void {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init("localhost", port);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void *in_ptr, *out_ptr;
        CUDA_ASSERT(cudaMalloc(&out_ptr, recvlen * sizeof(float)));

        if (rank == root) {
            CUDA_ASSERT(cudaMalloc(&in_ptr, nranks * recvlen * sizeof(float)));
            CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(),
                                   nranks * recvlen * sizeof(float),
                                   cudaMemcpyHostToDevice));
        } else {
            in_ptr = nullptr;
        }

        int ret = comm->scatter(in_ptr, out_ptr, recvlen,
                                MegRay::MEGRAY_FLOAT32, root, ctx);
        ASSERT_EQ(ret, 0);

        CUDA_ASSERT(cudaStreamSynchronize(stream));
        CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr,
                               recvlen * sizeof(float),
                               cudaMemcpyDeviceToHost));
    };
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs, run);
}

TEST(TestOpr, Gather) {
    const int nranks = 3;
    const size_t sendlen = 10;
    const int root = 1;

    std::vector<std::vector<float>> inputs(nranks);
    std::vector<std::vector<float>> outputs(nranks);
    for (size_t i = 0; i < nranks; i++) {
        for (size_t j = 0; j < sendlen; j++) {
            float val = 1.0 * (i + 1) * (j + 2);
            inputs[i].push_back(val);
            outputs[root].push_back(val);
        }
    }

    auto run = [nranks, sendlen, root](std::shared_ptr<MegRay::Communicator> comm,
                                       int port, int rank,
                                       std::vector<float>& input,
                                       std::vector<float>& output) -> void {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init("localhost", port);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void *in_ptr, *out_ptr;
        CUDA_ASSERT(cudaMalloc(&in_ptr, sendlen * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(),
                               sendlen * sizeof(float),
                               cudaMemcpyHostToDevice));

        if (rank == root) {
            CUDA_ASSERT(cudaMalloc(&out_ptr, nranks * sendlen * sizeof(float)));
        } else {
            out_ptr = nullptr;
        }

        int ret = comm->gather(in_ptr, out_ptr, sendlen,
                               MegRay::MEGRAY_FLOAT32, root, ctx);
        ASSERT_EQ(ret, 0);

        CUDA_ASSERT(cudaStreamSynchronize(stream));

        if (rank == root) {
            CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr,
                                   nranks * sendlen * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        }
    };
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs, run);
}

TEST(TestOpr, AllToAll) {
    const int nranks = 3;
    const size_t len = 6;

    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(nranks * len));
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(nranks * len));
    for (size_t i = 0; i < nranks; i++) {
        for (size_t j = 0; j < nranks; j++) {
            for (size_t k = 0; k < len; k++) {
                float val = 1.0 * (i + 1) * (j + 2) * (k + 3);
                inputs[i][j * len + k] = val;
                outputs[j][i * len + k] = val;
            }
        }
    }

    auto run = [nranks, len](std::shared_ptr<MegRay::Communicator> comm,
                             int port, int rank,
                             std::vector<float>& input,
                             std::vector<float>& output) -> void {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init("localhost", port);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void *in_ptr, *out_ptr;
        CUDA_ASSERT(cudaMalloc(&in_ptr, nranks * len * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(),
                               nranks * len * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMalloc(&out_ptr, nranks * len * sizeof(float)));

        int ret = comm->all_to_all(in_ptr, out_ptr, len,
                                   MegRay::MEGRAY_FLOAT32, ctx);
        ASSERT_EQ(ret, 0);

        CUDA_ASSERT(cudaStreamSynchronize(stream));

        CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr,
                               nranks * len * sizeof(float),
                               cudaMemcpyDeviceToHost));
    };
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs, run);
}

TEST(TestOpr, AllGather) {
    const int nranks = 3;
    const size_t sendlen = 10;

    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(sendlen));
    std::vector<std::vector<float>> outputs(
            nranks, std::vector<float>(nranks * sendlen));
    for (size_t j = 0; j < sendlen; j++) {
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            for (int k = 0; k < nranks; k++) {
                outputs[k][i * sendlen + j] = inputs[i][j];
            }
        }
    }

    auto run = [nranks, sendlen](std::shared_ptr<MegRay::Communicator> comm,
                                 int port, int rank,
                                 std::vector<float>& input,
                                 std::vector<float>& output) -> void {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init("localhost", port);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void *in_ptr, *out_ptr;
        CUDA_ASSERT(cudaMalloc(&in_ptr, sendlen * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&out_ptr, sendlen * nranks * sizeof(float)));

        CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(), sendlen * sizeof(float),
                               cudaMemcpyHostToDevice));

        int ret = comm->all_gather(in_ptr, out_ptr, sendlen,
                                   MegRay::MEGRAY_FLOAT32, ctx);
        ASSERT_EQ(ret, 0);

        CUDA_ASSERT(cudaStreamSynchronize(stream));
        CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr,
                               nranks * sendlen * sizeof(float),
                               cudaMemcpyDeviceToHost));
    };
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs, run);
}

TEST(TestOpr, AllReduce) {
    const int nranks = 3;
    const size_t len = 10;
    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(len));
    std::vector<std::vector<float>> expected_outputs(nranks,
                                                     std::vector<float>(len));

    auto reduce_func = [nranks, len](MegRay::ReduceOp op) {
        auto run = [nranks, len, op](std::shared_ptr<MegRay::Communicator> comm,
                                     int port, int rank,
                                     std::vector<float>& input,
                                     std::vector<float>& output) {
            CUDA_ASSERT(cudaSetDevice(rank));
            comm->init("localhost", port);

            cudaStream_t stream;
            CUDA_ASSERT(cudaStreamCreate(&stream));
            auto ctx = MegRay::CudaContext::make(stream);

            void *in_ptr, *out_ptr;
            CUDA_ASSERT(cudaMalloc(&in_ptr, len * sizeof(float)));
            CUDA_ASSERT(cudaMalloc(&out_ptr, len * sizeof(float)));

            CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(), len * sizeof(float),
                                   cudaMemcpyHostToDevice));

            int ret = comm->all_reduce(in_ptr, out_ptr, len,
                                       MegRay::MEGRAY_FLOAT32, op, ctx);
            ASSERT_EQ(ret, 0);

            CUDA_ASSERT(cudaStreamSynchronize(stream));
            CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr, len * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        };
        return run;
    };

    for (size_t j = 0; j < len; j++) {
        float sum = 0;
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            sum += inputs[i][j];
        }
        for (size_t i = 0; i < nranks; i++) {
            expected_outputs[i][j] = sum;
        }
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_SUM));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_SUM));

    for (size_t j = 0; j < len; j++) {
        float max_val = std::numeric_limits<float>::min();
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            max_val = std::max(max_val, inputs[i][j]);
        }
        for (size_t i = 0; i < nranks; i++) {
            expected_outputs[i][j] = max_val;
        }
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MAX));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MAX));

    for (size_t j = 0; j < len; j++) {
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            min_val = std::min(min_val, inputs[i][j]);
        }
        for (size_t i = 0; i < nranks; i++) {
            expected_outputs[i][j] = min_val;
        }
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MIN));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MIN));
}

TEST(TestOpr, ReduceScatterSum) {
    const int nranks = 3;
    const size_t recvlen = 10;

    std::vector<std::vector<float>> inputs(
            nranks, std::vector<float>(nranks * recvlen));
    std::vector<std::vector<float>> expected_outputs(
            nranks, std::vector<float>(recvlen));
    auto reduce_func = [nranks, recvlen](MegRay::ReduceOp op) {
        auto run = [nranks, recvlen,
                    op](std::shared_ptr<MegRay::Communicator> comm,
                        int port, int rank,
                        std::vector<float>& input, std::vector<float>& output) {
            CUDA_ASSERT(cudaSetDevice(rank));
            comm->init("localhost", port);

            cudaStream_t stream;
            CUDA_ASSERT(cudaStreamCreate(&stream));
            auto ctx = MegRay::CudaContext::make(stream);

            void *in_ptr, *out_ptr;
            CUDA_ASSERT(cudaMalloc(&in_ptr, nranks * recvlen * sizeof(float)));
            CUDA_ASSERT(cudaMalloc(&out_ptr, recvlen * sizeof(float)));

            CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(),
                                   nranks * recvlen * sizeof(float),
                                   cudaMemcpyHostToDevice));

            int ret = comm->reduce_scatter(in_ptr, out_ptr, recvlen,
                                           MegRay::MEGRAY_FLOAT32, op, ctx);
            ASSERT_EQ(ret, 0);

            CUDA_ASSERT(cudaStreamSynchronize(stream));
            CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr,
                                   recvlen * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        };
        return run;
    };

    for (int k = 0; k < nranks; k++) {
        for (size_t j = 0; j < recvlen; j++) {
            float sum = 0;
            for (size_t i = 0; i < nranks; i++) {
                int m = k * recvlen + j;
                inputs[i][m] = 1.0 * (i + 1) * (m + 1);
                sum += inputs[i][m];
            }
            expected_outputs[k][j] = sum;
        }
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_SUM));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_SUM));

    for (int k = 0; k < nranks; k++) {
        for (size_t j = 0; j < recvlen; j++) {
            float max_val = std::numeric_limits<float>::min();
            for (size_t i = 0; i < nranks; i++) {
                int m = k * recvlen + j;
                inputs[i][m] = 1.0 * (i + 1) * (m + 1);
                max_val = std::max(inputs[i][m], max_val);
            }
            expected_outputs[k][j] = max_val;
        }
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MAX));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MAX));
    for (int k = 0; k < nranks; k++) {
        for (size_t j = 0; j < recvlen; j++) {
            float min_val = std::numeric_limits<float>::max();
            for (size_t i = 0; i < nranks; i++) {
                int m = k * recvlen + j;
                inputs[i][m] = 1.0 * (i + 1) * (m + 1);
                min_val = std::min(inputs[i][m], min_val);
            }
            expected_outputs[k][j] = min_val;
        }
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MIN));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MIN));
}

TEST(TestOpr, Broadcast) {
    const int nranks = 3;
    const int root = 1;
    const size_t len = 10;

    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(len));
    std::vector<std::vector<float>> outputs(nranks, std::vector<float>(len));
    for (size_t j = 0; j < len; j++) {
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
        }
        for (size_t i = 0; i < nranks; i++) {
            outputs[i][j] = inputs[root][j];
        }
    }

    auto run = [nranks, root, len](std::shared_ptr<MegRay::Communicator> comm,
                                   int port, int rank,
                                   std::vector<float>& input,
                                   std::vector<float>& output) {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init("localhost", port);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void *in_ptr, *out_ptr;
        CUDA_ASSERT(cudaMalloc(&in_ptr, len * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&out_ptr, len * sizeof(float)));

        CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(), len * sizeof(float),
                               cudaMemcpyHostToDevice));

        int ret = comm->broadcast(in_ptr, out_ptr, len, MegRay::MEGRAY_FLOAT32,
                                  root, ctx);
        ASSERT_EQ(ret, 0);

        CUDA_ASSERT(cudaStreamSynchronize(stream));
        CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr, len * sizeof(float),
                               cudaMemcpyDeviceToHost));
    };

    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs, run);
}

TEST(TestOpr, ReduceSum) {
    const int nranks = 3;
    const int root = 1;
    const size_t len = 10;

    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(len));
    std::vector<std::vector<float>> expected_outputs(nranks);
    expected_outputs[root].resize(len);

    auto reduce_func = [nranks, root, len](MegRay::ReduceOp op) {
        auto run = [nranks, root, len,
                    op](std::shared_ptr<MegRay::Communicator> comm,
                        int port, int rank,
                        std::vector<float>& input, std::vector<float>& output) {
            CUDA_ASSERT(cudaSetDevice(rank));
            comm->init("localhost", port);

            cudaStream_t stream;
            CUDA_ASSERT(cudaStreamCreate(&stream));
            auto ctx = MegRay::CudaContext::make(stream);

            void *in_ptr, *out_ptr;
            CUDA_ASSERT(cudaMalloc(&in_ptr, len * sizeof(float)));
            if (rank == root) {
                CUDA_ASSERT(cudaMalloc(&out_ptr, len * sizeof(float)));
            }

            CUDA_ASSERT(cudaMemcpy(in_ptr, input.data(), len * sizeof(float),
                                   cudaMemcpyHostToDevice));

            int ret = comm->reduce(in_ptr, out_ptr, len, MegRay::MEGRAY_FLOAT32,
                                   op, root, ctx);
            ASSERT_EQ(ret, 0);

            CUDA_ASSERT(cudaStreamSynchronize(stream));
            if (rank == root) {
                CUDA_ASSERT(cudaMemcpy(output.data(), out_ptr,
                                       len * sizeof(float),
                                       cudaMemcpyDeviceToHost));
            }
        };
        return run;
    };
    for (size_t j = 0; j < len; j++) {
        float sum = 0;
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            sum += inputs[i][j];
        }
        expected_outputs[root][j] = sum;
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_SUM));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_SUM));
    for (size_t j = 0; j < len; j++) {
        float max_val = std::numeric_limits<float>::min();
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            max_val = std::max(inputs[i][j], max_val);
        }
        expected_outputs[root][j] = max_val;
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MAX));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MAX));
    for (size_t j = 0; j < len; j++) {
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            min_val = std::min(inputs[i][j], min_val);
        }
        expected_outputs[root][j] = min_val;
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MIN));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    reduce_func(MegRay::MEGRAY_MIN));
}
