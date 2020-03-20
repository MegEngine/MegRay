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

#include "../src/megray.h"
#include "test_base.h"

TEST(TestNcclCommunicator, Init) {
    const int nranks = 3;

    std::vector<std::shared_ptr<MegRay::Communicator>> comms(nranks);
    std::vector<std::string> uids(nranks);
    for (size_t i = 0; i < nranks; i++) {
        comms[i] = MegRay::get_communicator(nranks, i, MegRay::MEGRAY_NCCL);
        uids[i] = comms[i]->get_uid();
    }

    auto run = [&](int rank) { comms[rank]->init(uids); };

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

    std::vector<std::shared_ptr<MegRay::Communicator>> comms(nranks);
    std::vector<std::string> uids(nranks);
    for (int i = 0; i < nranks; i++) {
        comms[i] = MegRay::get_communicator(nranks, i, MegRay::MEGRAY_UCX);
        uids[i] = comms[i]->get_uid();
    }

    auto run = [&](int rank) {
        cudaSetDevice(rank);
        comms[rank]->init(uids);
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
    auto send_comm = MegRay::get_communicator(2, 0, MegRay::MEGRAY_UCX);
    auto recv_comm = MegRay::get_communicator(2, 1, MegRay::MEGRAY_UCX);

    std::vector<std::string> uids(2);
    uids[0] = send_comm->get_uid();
    uids[1] = recv_comm->get_uid();

    std::string msg("test_message");
    size_t len = msg.size();
    std::string output;

    auto sender = [&]() {
        CUDA_ASSERT(cudaSetDevice(0));
        send_comm->init(uids);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void* ptr;
        cudaMalloc(&ptr, len);
        CUDA_ASSERT(cudaMemcpy(ptr, msg.data(), len, cudaMemcpyHostToDevice));

        send_comm->send(msg.data(), len, 1, ctx);

        CUDA_ASSERT(cudaStreamSynchronize(stream));
    };

    auto receiver = [&]() {
        CUDA_ASSERT(cudaSetDevice(1));
        recv_comm->init(uids);

        cudaStream_t stream;
        CUDA_ASSERT(cudaStreamCreate(&stream));
        auto ctx = MegRay::CudaContext::make(stream);

        void* ptr;
        CUDA_ASSERT(cudaMalloc(&ptr, len));
        recv_comm->recv(ptr, len, 0, ctx);

        CUDA_ASSERT(cudaStreamSynchronize(stream));

        char* outbuff = new char[len];
        CUDA_ASSERT(cudaMemcpy(outbuff, ptr, len, cudaMemcpyDeviceToHost));
        output = std::string(outbuff, len);
        delete outbuff;
    };

    std::thread send_th(sender);
    std::thread recv_th(receiver);

    send_th.join();
    recv_th.join();

    ASSERT_EQ(msg, output);
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
                                 std::vector<std::string>& uids, int rank,
                                 std::vector<float>& input,
                                 std::vector<float>& output) -> void {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init(uids);

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
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs,
                    MegRay::MEGRAY_FLOAT32, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs,
                    MegRay::MEGRAY_FLOAT32, run);
}

TEST(TestOpr, AllReduce) {
    const int nranks = 3;
    const size_t len = 10;
    std::vector<std::vector<float>> inputs(nranks, std::vector<float>(len));
    std::vector<std::vector<float>> expected_outputs(nranks,
                                                     std::vector<float>(len));

    auto reduce_func = [nranks, len](MegRay::ReduceOp op) {
        auto run = [nranks, len, op](std::shared_ptr<MegRay::Communicator> comm,
                                     std::vector<std::string>& uids, int rank,
                                     std::vector<float>& input,
                                     std::vector<float>& output) {
            CUDA_ASSERT(cudaSetDevice(rank));
            comm->init(uids);

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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_SUM));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_SUM));

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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MAX));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MAX));

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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MIN));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MIN));
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
                        std::vector<std::string>& uids, int rank,
                        std::vector<float>& input, std::vector<float>& output) {
            CUDA_ASSERT(cudaSetDevice(rank));
            comm->init(uids);

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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_SUM));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_SUM));

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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MAX));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MAX));
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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MIN));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MIN));
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
                                   std::vector<std::string>& uids, int rank,
                                   std::vector<float>& input,
                                   std::vector<float>& output) {
        CUDA_ASSERT(cudaSetDevice(rank));
        comm->init(uids);

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

    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, outputs,
                    MegRay::MEGRAY_FLOAT32, run);
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, outputs,
                    MegRay::MEGRAY_FLOAT32, run);
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
                        std::vector<std::string>& uids, int rank,
                        std::vector<float>& input, std::vector<float>& output) {
            CUDA_ASSERT(cudaSetDevice(rank));
            comm->init(uids);

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
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_SUM));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_SUM));
    for (size_t j = 0; j < len; j++) {
        float max_val = std::numeric_limits<float>::min();
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            max_val = std::max(inputs[i][j], max_val);
        }
        expected_outputs[root][j] = max_val;
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MAX));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MAX));
    for (size_t j = 0; j < len; j++) {
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < nranks; i++) {
            inputs[i][j] = 1.0 * (i + 1) * (j + 1);
            min_val = std::min(inputs[i][j], min_val);
        }
        expected_outputs[root][j] = min_val;
    }
    run_test<float>(nranks, MegRay::MEGRAY_NCCL, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MIN));
    run_test<float>(nranks, MegRay::MEGRAY_UCX, inputs, expected_outputs,
                    MegRay::MEGRAY_FLOAT32, reduce_func(MegRay::MEGRAY_MIN));
}
