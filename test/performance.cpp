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

#include "args.hxx"
#include "megray.h"
#include "test_utils.h"

struct Arguments {
    // global arguments
    MegRay::Backend backends;  // MEGRAY_BACKEND_COUNT means ALL
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
    if (args.backends == MegRay::MEGRAY_BACKEND_COUNT ||
        args.backends == MegRay::MEGRAY_NCCL)
        backends.emplace_back("NCCL", MegRay::MEGRAY_NCCL);
#endif
#ifdef MEGRAY_WITH_UCX
    if (args.backends == MegRay::MEGRAY_BACKEND_COUNT ||
        args.backends == MegRay::MEGRAY_UCX)
        backends.emplace_back("UCX", MegRay::MEGRAY_UCX);
#endif
#ifdef MEGRAY_WITH_RCCL
    if (args.backends == MegRay::MEGRAY_BACKEND_COUNT ||
        args.backends == MegRay::MEGRAY_RCCL)
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

int main(int argc, char* argv[]) {
    // args definition
    args::ArgumentParser parser("This is a benchmark for MegRay.",
                                "local example: ./performance -d 4\n\n"
                                "distributed example1: ./performance -d 4 "
                                "--ip=127.0.0.1 -n 2 -r 0\n"
                                "distributed example2: ./performance -d 4 "
                                "--ip=127.0.0.1 -n 2 -r 1");
    parser.helpParams.addDefault = true;
    parser.helpParams.addChoices = true;
    args::HelpFlag help(parser, "help", "Display this help menu",
                        {'h', "help"});
    args::Group commands(parser, "compulsive", args::Group::Validators::All);
    args::ValueFlag<int> n_devs(commands, "int", "Number of devs",
                                {'d', "devs"});

    args::Group options(parser, "optional");
    args::ValueFlag<std::string> master_ip(options, "string", "Master IP",
                                           {'I', "ip"}, "127.0.0.1");
    args::ValueFlag<int> port(options, "int", "Master port", {'P', "port"},
                              2395);
    args::ValueFlag<int> n_nodes(options, "int", "Number of nodes",
                                 {'n', "nnodes"}, 1);
    args::ValueFlag<int> node_rank(options, "int", "Rank of node",
                                   {'r', "node"}, 0);
    args::MapFlag<std::string, MegRay::Backend> backend(
            options, "string", "Choose a backend for communication",
            {'e', "backend"},
            {{"ALL", MegRay::MEGRAY_BACKEND_COUNT},
             {"NCCL", MegRay::MEGRAY_NCCL},
             {"UCX", MegRay::MEGRAY_UCX},
             {"RCCL", MegRay::MEGRAY_RCCL}},
            MegRay::MEGRAY_BACKEND_COUNT);
    args::ValueFlag<std::string> func_select(
            options, "string",
            "Select function: ALL, all_reduce, send_recv, scatter, gather, "
            "all_to_all, all_gather, reduce_scatter, broadcast, reduce",
            {'f', "func"}, "ALL");
    args::ValueFlagList<int> sizes(
            options, "int", "packet size", {'s', "size"},
            {8, 1024, 512 * 1024, 1024 * 1024, 10 * 1024 * 1024});
    args::ValueFlag<int> warmup_iters(options, "int", "warmup iterations",
                                      {'w', "warmup"}, 100);
    args::ValueFlag<int> iters(options, "int", "test iterations", {'i', "iter"},
                               100);
    args::ValueFlag<int> send_rank(
            options, "int", "send/recv operation send node rank", {"srank"}, 0);
    args::ValueFlag<int> recv_rank(
            options, "int", "send/recv operation recv node rank", {"rrank"}, 0);
    args::ValueFlag<int> send_dev(
            options, "int", "send/recv operation send node dev", {"sdev"}, 0);
    args::ValueFlag<int> recv_dev(
            options, "int", "send/recv operation recv node dev", {"rdev"}, 1);

    // args parser and validation
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    } catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    Arguments args = {
            args::get(backend),   args::get(master_ip),
            args::get(port),      args::get(n_nodes),
            args::get(node_rank), args::get(n_devs),
            args::get(sizes),     args::get(send_rank),
            args::get(send_dev),  args::get(recv_rank),
            args::get(recv_dev),  args::get(warmup_iters),
            args::get(iters),     args::get(func_select),
    };
    init_maps(args);

    main_test(args);

    return 0;
}
