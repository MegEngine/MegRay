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
#include <unistd.h>

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
    // global arguments
    std::string backends;
    std::string master_ip;
    int port;
    int n_nodes;
    int node_rank;
    int n_devs;
    std::vector<int> sizes;
    int send_rank;
    int send_dev;
    int recv_rank;
    int recv_dev;
    int warmup_iters;
    int iters;
    std::string func_select;

    // worker arguments
    MegRay::Backend backend;
    size_t bufsize;
    size_t in_bufsize;
    size_t out_bufsize;

    Arguments() {
        backends = "ALL";
        master_ip = "127.0.0.1";
        port = 2395;
        n_nodes = 1;
        node_rank = 0;
        n_devs = 2;
        sizes = {8, 1024, 512 * 1024, 1024 * 1024, 10 * 1024 * 1024};
        send_rank = 0;
        send_dev = 0;
        recv_rank = 0;
        recv_dev = 1;
        warmup_iters = 100;
        iters = 100;
        func_select = "ALL";
    }
};

typedef std::function<void(Arguments, int, size_t*)> WorkerFunc;

typedef std::function<void(std::shared_ptr<MegRay::Communicator>, void*, void*,
                           std::shared_ptr<MegRay::Context>)>
        KernelFunc;

std::vector<std::pair<std::string, MegRay::Backend>> backends;
std::vector<std::pair<std::string, WorkerFunc>> funcs;

size_t get_timestamp() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}

size_t run_test(WorkerFunc func, Arguments args) {
    size_t res = 0;
    int world_size = args.n_nodes * args.n_devs;

    MegRay::create_server(world_size, args.port);

    std::vector<std::thread> threads;
    for (int i = 0; i < args.n_devs; i++) {
        int global_rank = args.node_rank * args.n_devs + i;
        threads.push_back(std::thread(func, args, i, &res));
    }

    for (int i = 0; i < args.n_devs; i++) {
        threads[i].join();
    }
    return res;
}

void worker(KernelFunc kernel, Arguments args, int dev, size_t* res) {
    int world_size = args.n_nodes * args.n_devs;
    int global_rank = args.node_rank * args.n_devs + dev;

    auto trait = get_context_trait(get_preferred_context(args.backend));
    trait.set_device(dev);
    auto context = trait.make_context();

    auto comm = MegRay::get_communicator(world_size, global_rank, args.backend);
    comm->init(args.master_ip.c_str(), args.port);

    void* in_ptr = trait.alloc(args.in_bufsize);
    void* out_ptr = trait.alloc(args.out_bufsize);

    for (size_t i = 0; i < args.warmup_iters; i++) {
        kernel(comm, in_ptr, out_ptr, context);
    }

    trait.sync_context(context);
    size_t t0 = get_timestamp();

    for (size_t i = 0; i < args.iters; i++) {
        kernel(comm, in_ptr, out_ptr, context);
    }

    trait.sync_context(context);
    size_t t1 = get_timestamp();

    trait.free(in_ptr);
    trait.free(out_ptr);

    if (dev == 0) {
        *res = (t1 - t0) / args.iters;
    }
}

void run_send_recv(Arguments args, int dev, size_t* res) {
    auto kernel = [args, dev](std::shared_ptr<MegRay::Communicator> comm,
                              void* in_ptr, void* out_ptr,
                              std::shared_ptr<MegRay::Context> context) {
        if (args.node_rank == args.send_rank and dev == args.send_dev) {
            comm->send(in_ptr, args.bufsize / sizeof(float),
                       MegRay::MEGRAY_FLOAT32,
                       args.recv_rank * args.n_devs + args.recv_dev, context);
        } else if (args.node_rank == args.recv_rank and dev == args.recv_dev) {
            comm->recv(out_ptr, args.bufsize / sizeof(float),
                       MegRay::MEGRAY_FLOAT32,
                       args.send_rank * args.n_devs + args.send_dev, context);
        }
    };
    args.in_bufsize = args.bufsize;
    args.out_bufsize = args.bufsize;
    worker(kernel, args, dev, res);
}

void run_scatter(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->scatter(in_ptr, out_ptr, args.bufsize / sizeof(float),
                      MegRay::MEGRAY_FLOAT32, 0, context);
    };
    args.in_bufsize = args.bufsize * args.n_nodes * args.n_devs;
    args.out_bufsize = args.bufsize;
    worker(kernel, args, dev, res);
}

void run_gather(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->gather(in_ptr, out_ptr, args.bufsize / sizeof(float),
                     MegRay::MEGRAY_FLOAT32, 0, context);
    };
    args.in_bufsize = args.bufsize;
    args.out_bufsize = args.bufsize * args.n_nodes * args.n_devs;
    worker(kernel, args, dev, res);
}

void run_all_to_all(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->all_to_all(in_ptr, out_ptr, args.bufsize / sizeof(float),
                         MegRay::MEGRAY_FLOAT32, context);
    };
    args.in_bufsize = args.bufsize * args.n_nodes * args.n_devs;
    args.out_bufsize = args.in_bufsize;
    worker(kernel, args, dev, res);
}

void run_all_gather(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->all_gather(in_ptr, out_ptr, args.bufsize / sizeof(float),
                         MegRay::MEGRAY_FLOAT32, context);
    };
    args.in_bufsize = args.bufsize;
    args.out_bufsize = args.bufsize * args.n_nodes * args.n_devs;
    worker(kernel, args, dev, res);
}

void run_all_reduce_sum(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->all_reduce(in_ptr, out_ptr, args.bufsize / sizeof(float),
                         MegRay::MEGRAY_FLOAT32, MegRay::MEGRAY_SUM, context);
    };
    args.in_bufsize = args.bufsize;
    args.out_bufsize = args.bufsize;
    worker(kernel, args, dev, res);
}

void run_reduce_scatter_sum(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->reduce_scatter(in_ptr, out_ptr, args.bufsize / sizeof(float),
                             MegRay::MEGRAY_FLOAT32, MegRay::MEGRAY_SUM,
                             context);
    };
    args.in_bufsize = args.bufsize * args.n_nodes * args.n_devs;
    args.out_bufsize = args.bufsize;
    worker(kernel, args, dev, res);
}

void run_broadcast(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->broadcast(in_ptr, out_ptr, args.bufsize / sizeof(float),
                        MegRay::MEGRAY_FLOAT32, 0, context);
    };
    args.in_bufsize = args.bufsize;
    args.out_bufsize = args.bufsize;
    worker(kernel, args, dev, res);
}

void run_reduce_sum(Arguments args, int dev, size_t* res) {
    auto kernel = [args](std::shared_ptr<MegRay::Communicator> comm,
                         void* in_ptr, void* out_ptr,
                         std::shared_ptr<MegRay::Context> context) {
        comm->reduce(in_ptr, out_ptr, args.bufsize / sizeof(float),
                     MegRay::MEGRAY_FLOAT32, MegRay::MEGRAY_SUM, 0, context);
    };
    args.in_bufsize = args.bufsize;
    args.out_bufsize = args.bufsize;
    worker(kernel, args, dev, res);
}

void init_maps(Arguments args) {
#ifdef MEGRAY_WITH_NCCL
    if (args.backends == "ALL" || args.backends == "NCCL")
        backends.emplace_back("NCCL", MegRay::MEGRAY_NCCL);
#endif
#ifdef MEGRAY_WITH_UCX
    if (args.backends == "ALL" || args.backends == "UCX")
        backends.emplace_back("UCX", MegRay::MEGRAY_UCX);
#endif
#ifdef MEGRAY_WITH_RCCL
    if (args.backends == "ALL" || args.backends == "RCCL")
        backends.emplace_back("RCCL", MegRay::MEGRAY_RCCL);
#endif

    if (args.func_select == "ALL") {
        funcs.emplace_back("send_recv", run_send_recv);
        funcs.emplace_back("all_reduce_sum", run_all_reduce_sum);
        funcs.emplace_back("scatter", run_scatter);
        funcs.emplace_back("gather", run_gather);
        funcs.emplace_back("all_to_all", run_all_to_all);
        funcs.emplace_back("all_gather", run_all_gather);
        funcs.emplace_back("reduce_scatter_sum", run_reduce_scatter_sum);
        funcs.emplace_back("broadcast", run_broadcast);
        funcs.emplace_back("reduce_sum", run_reduce_sum);
    } else if (args.func_select == "all_reduce") {
        funcs.emplace_back("all_reduce_sum", run_all_reduce_sum);
    } else if (args.func_select == "send_recv") {
        funcs.emplace_back("send_recv", run_send_recv);
    } else if (args.func_select == "scatter") {
        funcs.emplace_back("scatter", run_scatter);
    } else if (args.func_select == "gather") {
        funcs.emplace_back("gather", run_gather);
    } else if (args.func_select == "all_to_all") {
        funcs.emplace_back("all_to_all", run_all_to_all);
    } else if (args.func_select == "all_gather") {
        funcs.emplace_back("all_gather", run_all_gather);
    } else if (args.func_select == "reduce_scatter") {
        funcs.emplace_back("reduce_scatter_sum", run_reduce_scatter_sum);
    } else if (args.func_select == "broadcast") {
        funcs.emplace_back("broadcast", run_broadcast);
    } else if (args.func_select == "reduce") {
        funcs.emplace_back("reduce_sum", run_reduce_sum);
    } else {
        std::cerr << "ERROR argument func: " << args.func_select << std::endl;
        exit(1);
    }
}

template <typename T>
void print(T n) {
    std::cout << std::right << std::setw(13) << std::setfill(' ') << n;
}

void print(float f) {
    std::cout << std::right << std::setw(13) << std::setfill(' ')
              << std::setprecision(4) << std::fixed << f;
}

int main_test(Arguments args) {
    std::cout << "Running on " << args.n_nodes << " nodes "
              << args.n_nodes * args.n_devs << " GPUs" << std::endl;
    std::cout << "Master address " << args.master_ip << ":" << args.port
              << std::endl;
    std::cout << "sizes: ";
    for (auto& s : args.sizes) {
        std::cout << s << ", ";
    }
    std::cout << std::endl;

    if (args.node_rank == 0) {
        MegRay::create_server(args.n_nodes, args.port);
    }

    auto client =
            std::make_unique<MegRay::Client>(args.n_nodes, args.node_rank);
    auto res = client->connect(args.master_ip.c_str(), args.port);
    client->barrier();

    for (auto f_it : funcs) {
        for (auto b_it : backends) {
            std::cout << std::endl
                      << "Testing " << f_it.first << " on backend "
                      << b_it.first << std::endl;
            print("bufsize");
            print("time(us)");
            print("speed(GB/s)");
            print("iters");
            std::cout << std::endl;

            for (auto& s_it : args.sizes) {
                int test_port = MegRay::get_free_port();
                client->broadcast(&test_port, &test_port, sizeof(int), 0);

                args.port = test_port;
                args.backend = b_it.second;
                args.bufsize = s_it;

                size_t avg_time = run_test(f_it.second, args);
                float speed = 1.0 * s_it / avg_time / 1000;

                print(s_it);
                print(avg_time);
                print(speed);
                print(args.iters);
                std::cout << std::endl;
            }
        }
    }
    return 0;
}

void show_help() {
    std::cout
            << "./performance {OPTIONS}\n"
               "\n"
               "    This is a benchmark for MegRay.\n"
               "\n"
               "  OPTIONS:\n"
               "\n"
               "      -h                                Display this help "
               "menu\n"
               "\n"
               "      optional\n"
               "        -d[int]                           Number of devs\n"
               "                                            Default: 2\n"
               "        -I[string]                        Master IP\n"
               "                                            Default: "
               "127.0.0.1\n"
               "        -P[int]                           Master port\n"
               "                                            Default: 2395\n"
               "        -n[int]                           Number of nodes\n"
               "                                            Default: 1\n"
               "        -r[int]                           Rank of node\n"
               "                                            Default: 0\n"
               "        -e[string]                        Choose a backend for "
               "communication\n"
               "                                          One of: ALL, NCCL, "
               "RCCL, UCX\n"
               "        -f[string]                        Select function: "
               "ALL, all_reduce,\n"
               "                                          send_recv, scatter, "
               "gather,\n"
               "                                          all_to_all, "
               "all_gather,\n"
               "                                          reduce_scatter, "
               "broadcast, reduce\n"
               "                                            Default: ALL\n"
               "        -s[int]                           packet size\n"
               "        -w[int]                           warmup iterations\n"
               "                                            Default: 100\n"
               "        -i[int]                           test iterations\n"
               "                                            Default: 100\n"
               "        -A=[int]                          send/recv operation "
               "send node rank\n"
               "                                            Default: 0\n"
               "        -B=[int]                          send/recv operation "
               "recv node rank\n"
               "                                            Default: 0\n"
               "        -C=[int]                          send/recv operation "
               "send node dev\n"
               "                                            Default: 0\n"
               "        -D=[int]                          send/recv operation "
               "recv node dev\n"
               "                                            Default: 1\n"
               "\n"
               "    local example: ./performance -d 4\n"
               "    distributed example1: ./performance -d 4 -I 127.0.0.1 -n "
               "2 -r 0\n"
               "    distributed example2: ./performance -d 4 -I 127.0.0.1 -n "
               "2 -r 1\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
    // args definition
    Arguments args;

    char ch;
    while ((ch = getopt(argc, argv, "hd:I:P:n:r:e:f:s:w:i:A:B:C:D:")) != EOF) {
        switch (ch) {
            case 'h':
                show_help();
                exit(0);
            case 'd':
                args.n_devs = atoi(optarg);
                break;
            case 'I':
                args.master_ip = optarg;
                break;
            case 'P':
                args.port = atoi(optarg);
                break;
            case 'n':
                args.n_nodes = atoi(optarg);
                break;
            case 'r':
                args.node_rank = atoi(optarg);
                break;
            case 'e':
                args.backends = optarg;
                break;
            case 'f':
                args.func_select = optarg;
                break;
            case 's':
                args.sizes = std::vector<int>(1, atoi(optarg));
                break;
            case 'w':
                args.warmup_iters = atoi(optarg);
                break;
            case 'i':
                args.iters = atoi(optarg);
                break;
            case 'A':
                args.send_rank = atoi(optarg);
                break;
            case 'B':
                args.send_dev = atoi(optarg);
                break;
            case 'C':
                args.recv_rank = atoi(optarg);
                break;
            case 'D':
                args.recv_dev = atoi(optarg);
                break;
            default:
                std::cerr << "Error arguments" << std::endl;
                show_help();
                exit(1);
        }
    }

    // args parser and validation
    init_maps(args);

    // run
    main_test(args);

    return 0;
}
