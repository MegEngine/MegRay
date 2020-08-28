/**
 * \file test/performance.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "megray.h"
#include "test_utils.h"

struct Arguments {
    MegRay::Backend backend;
    char* master_ip;
    int port;
    int n_nodes;
    int node_rank;
    int n_devs;
    size_t in_bufsize;
    size_t out_bufsize;
};

typedef std::function<void(Arguments, int, size_t, size_t*)> WorkerFunc;

typedef std::function<void(std::shared_ptr<MegRay::Communicator>, void*, void*,
                           std::shared_ptr<MegRay::Context>)>
        KernelFunc;

std::map<std::string, MegRay::Backend> backends;
std::map<std::string, size_t> bufsizes;
std::map<std::string, WorkerFunc> funcs;

size_t get_timestamp() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}

size_t run_test(WorkerFunc func, Arguments args, size_t bufsize) {
    size_t res = 0;
    int world_size = args.n_nodes * args.n_devs;

    MegRay::create_server(world_size, args.port);

    std::vector<std::thread> threads;
    for (int i = 0; i < args.n_devs; i++) {
        int global_rank = args.node_rank * args.n_devs + i;
        threads.push_back(std::thread(func, args, i, bufsize, &res));
    }

    for (int i = 0; i < args.n_devs; i++) {
        threads[i].join();
    }
    return res;
}

void worker(KernelFunc kernel, Arguments args, int dev, size_t* res) {
    const int warmup_iters = 100;
    const int iters = 100;

    int world_size = args.n_nodes * args.n_devs;
    int global_rank = args.node_rank * args.n_devs + dev;

    auto trait = get_context_trait(get_preferred_context(args.backend));
    trait.set_device(dev);
    auto context = trait.make_context();

    auto comm = MegRay::get_communicator(world_size, global_rank, args.backend);
    comm->init(args.master_ip, args.port);

    void* in_ptr = trait.alloc(args.in_bufsize);
    void* out_ptr = trait.alloc(args.out_bufsize);

    for (size_t i = 0; i < warmup_iters; i++) {
        kernel(comm, in_ptr, out_ptr, context);
    }

    trait.sync_context(context);
    size_t t0 = get_timestamp();

    for (size_t i = 0; i < iters; i++) {
        kernel(comm, in_ptr, out_ptr, context);
    }

    trait.sync_context(context);
    size_t t1 = get_timestamp();

    trait.free(in_ptr);
    trait.free(out_ptr);

    if (dev == 0) {
        *res = (t1 - t0) / iters;
    }
}

void run_send_recv(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [args, bufsize, dev](
                          std::shared_ptr<MegRay::Communicator> comm,
                          void* in_ptr, void* out_ptr,
                          std::shared_ptr<MegRay::Context> context) {
        if (args.node_rank == 0 and dev == 0) {
            comm->send(in_ptr, args.in_bufsize / sizeof(float), 1, context);
        } else if (args.node_rank == 0 and dev == 1) {
            comm->recv(out_ptr, args.out_bufsize / sizeof(float), 0, context);
        }
    };
    args.in_bufsize = bufsize;
    args.out_bufsize = bufsize;
    worker(kernel, args, dev, res);
}

void run_scatter(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->scatter(in_ptr, out_ptr, bufsize / sizeof(float),
                      MegRay::MEGRAY_FLOAT32, 0, context);
    };
    args.in_bufsize = bufsize * args.n_nodes * args.n_devs;
    args.out_bufsize = bufsize;
    worker(kernel, args, dev, res);
}

void run_gather(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->gather(in_ptr, out_ptr, bufsize / sizeof(float),
                     MegRay::MEGRAY_FLOAT32, 0, context);
    };
    args.in_bufsize = bufsize;
    args.out_bufsize = bufsize * args.n_nodes * args.n_devs;
    worker(kernel, args, dev, res);
}

void run_all_to_all(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->all_to_all(in_ptr, out_ptr, bufsize / sizeof(float),
                         MegRay::MEGRAY_FLOAT32, context);
    };
    args.in_bufsize = bufsize * args.n_nodes * args.n_devs;
    args.out_bufsize = args.in_bufsize;
    worker(kernel, args, dev, res);
}

void run_all_gather(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->all_gather(in_ptr, out_ptr, bufsize / sizeof(float),
                         MegRay::MEGRAY_FLOAT32, context);
    };
    args.in_bufsize = bufsize;
    args.out_bufsize = bufsize * args.n_nodes * args.n_devs;
    worker(kernel, args, dev, res);
}

void run_all_reduce_sum(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->all_reduce(in_ptr, out_ptr, bufsize / sizeof(float),
                         MegRay::MEGRAY_FLOAT32, MegRay::MEGRAY_SUM, context);
    };
    args.in_bufsize = bufsize;
    args.out_bufsize = bufsize;
    worker(kernel, args, dev, res);
}

void run_reduce_scatter_sum(Arguments args, int dev, size_t bufsize,
                            size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->reduce_scatter(in_ptr, out_ptr, bufsize / sizeof(float),
                             MegRay::MEGRAY_FLOAT32, MegRay::MEGRAY_SUM,
                             context);
    };
    args.in_bufsize = bufsize * args.n_nodes * args.n_devs;
    args.out_bufsize = bufsize;
    worker(kernel, args, dev, res);
}

void run_broadcast(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->broadcast(in_ptr, out_ptr, bufsize / sizeof(float),
                        MegRay::MEGRAY_FLOAT32, 0, context);
    };
    args.in_bufsize = bufsize;
    args.out_bufsize = bufsize;
    worker(kernel, args, dev, res);
}

void run_reduce_sum(Arguments args, int dev, size_t bufsize, size_t* res) {
    auto kernel = [bufsize](std::shared_ptr<MegRay::Communicator> comm,
                            void* in_ptr, void* out_ptr,
                            std::shared_ptr<MegRay::Context> context) {
        comm->reduce(in_ptr, out_ptr, bufsize / sizeof(float),
                     MegRay::MEGRAY_FLOAT32, MegRay::MEGRAY_SUM, 0, context);
    };
    args.in_bufsize = bufsize;
    args.out_bufsize = bufsize;
    worker(kernel, args, dev, res);
}

void init_maps() {
#ifdef MEGRAY_WITH_NCCL
    backends["NCCL"] = MegRay::MEGRAY_NCCL;
#endif
#ifdef MEGRAY_WITH_UCX
    backends["UCX"] = MegRay::MEGRAY_UCX;
#endif
#ifdef MEGRAY_WITH_RCCL
    backends["RCCL"] = MegRay::MEGRAY_RCCL;
#endif

    bufsizes["  8B"] = 8;
    bufsizes[" 1KB"] = 1024;
    bufsizes[" 1MB"] = 1024 * 1024;
    bufsizes["10MB"] = 10 * 1024 * 1024;

    funcs["send_recv"] = run_send_recv;
    funcs["scatter"] = run_scatter;
    funcs["gather"] = run_gather;
    funcs["all_to_all"] = run_all_to_all;
    funcs["all_gather"] = run_all_gather;
    funcs["all_reduce_sum"] = run_all_reduce_sum;
    funcs["reduce_scatter_sum"] = run_reduce_scatter_sum;
    funcs["broadcast"] = run_broadcast;
    funcs["reduce_sum"] = run_reduce_sum;
}

template <typename T>
void print(T n) {
    std::cout << std::right << std::setw(13) << std::setfill(' ') << n;
}

void print(float f) {
    std::cout << std::right << std::setw(13) << std::setfill(' ')
              << std::setprecision(4) << std::fixed << f;
}

void print_help() {
    std::cout << "./performance <master_ip> <port> <n_nodes> <node_rank> "
                 "<n_devs_per_node>"
              << std::endl
              << std::endl;
    std::cout << "Example (4 nodes 32 gpus):" << std::endl;
    std::cout << "    ./performance 10.160.36.120 9000 4 0 8" << std::endl;
    std::cout << "    ./performance 10.160.36.120 9000 4 1 8" << std::endl;
    std::cout << "    ./performance 10.160.36.120 9000 4 2 8" << std::endl;
    std::cout << "    ./performance 10.160.36.120 9000 4 3 8" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        print_help();
        return -1;
    }

    char* master_ip = argv[1];
    if (strlen(master_ip) < 7) {
        print_help();
        std::cout << std::endl << "Invalid master_ip: " << argv[1] << std::endl;
    }

    int port = std::atoi(argv[2]);
    if (port <= 0) {
        print_help();
        std::cout << std::endl << "Invalid port: " << argv[2] << std::endl;
        return -1;
    }

    int n_nodes = std::atoi(argv[3]);
    if (n_nodes <= 0) {
        print_help();
        std::cout << std::endl << "Invalid n_nodes: " << argv[3] << std::endl;
        return -1;
    }

    int node_rank = std::atoi(argv[4]);
    if (node_rank < 0 or node_rank >= n_nodes) {
        print_help();
        std::cout << std::endl << "Invalid node_rank: " << argv[4] << std::endl;
        return -1;
    }

    int n_devs = std::atoi(argv[5]);
    if (n_devs <= 0 or n_devs > 8) {
        print_help();
        std::cout << std::endl << "Invalid n_devs: " << argv[5] << std::endl;
        return -1;
    }

    std::cout << "Running on " << n_nodes << " nodes " << n_nodes * n_devs
              << " GPUs" << std::endl;
    std::cout << "Master address " << master_ip << ":" << port << std::endl;

    init_maps();

    if (node_rank == 0) {
        MegRay::create_server(n_nodes, port);
    }

    auto client = std::make_unique<MegRay::Client>(n_nodes, node_rank);
    auto res = client->connect(master_ip, port);
    client->barrier();

    for (auto b_it : backends) {
        for (auto f_it : funcs) {
            std::cout << std::endl
                      << "Testing " << f_it.first << " on backend "
                      << b_it.first << std::endl;
            print("bufsize");
            print("time(us)");
            print("speed(GB/s)");
            std::cout << std::endl;

            for (auto s_it : bufsizes) {
                int test_port = MegRay::get_free_port();
                client->broadcast(&test_port, &test_port, sizeof(int), 0);

                Arguments args = {b_it.second, master_ip, test_port, n_nodes,
                                  node_rank,   n_devs,    0,         0};
                size_t avg_time = run_test(f_it.second, args, s_it.second);
                float speed = 1.0 * s_it.second / avg_time / 1000;

                print(s_it.first);
                print(avg_time);
                print(speed);
                std::cout << std::endl;
            }
        }
    }

    return 0;
}
