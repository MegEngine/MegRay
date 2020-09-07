/**
 * \file src/ucx/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "communicator.h"

#include <cstring>

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

// returned handler of ucp requests
// flag "completed" is set in callback functions
struct Request {
    int completed;
};

// init request handler
static void request_init(void* request) {
    static_cast<Request*>(request)->completed = 0;
}

// send callback, set flag "completed" to 1
static void send_cb(void* request, ucs_status_t status) {
    ((Request*)request)->completed = 1;
}

// receive callback, set flag "completed" to 1
static void recv_cb(void* request, ucs_status_t status,
                    ucp_tag_recv_info_t* info) {
    ((Request*)request)->completed = 1;
}

UcxCommunicator::UcxCommunicator(int nranks, int rank)
        : Communicator(nranks, rank), m_inited(false) {
    const char* env = "UCX_MEMTYPE_CACHE=n";
    putenv((char*)env);

    // set ucp context params
    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                            UCP_PARAM_FIELD_REQUEST_SIZE |
                            UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features =
            UCP_FEATURE_TAG | UCP_FEATURE_RMA | UCP_FEATURE_WAKEUP;
    ucp_params.request_size = sizeof(Request);
    ucp_params.request_init = request_init;

    // init ucp context
    ucs_status_t status;
    status = ucp_init(&ucp_params, nullptr, &m_context);
    MEGRAY_ASSERT(status == UCS_OK, "failed to init ucp context");

    // set ucp worker params
    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    // create ucp worker
    status = ucp_worker_create(m_context, &worker_params, &m_worker);
    MEGRAY_ASSERT(status == UCS_OK, "failed to create ucp worker");
}

UcxCommunicator::~UcxCommunicator() {
    // destroy ucp worker and cleanup ucp context
    ucp_worker_destroy(m_worker);
    ucp_cleanup(m_context);
}

Status UcxCommunicator::do_init() {
    // get ucp worker address
    size_t addr_len, addr_lens[m_nranks];
    ucp_address_t* addr;
    ucs_status_t status = ucp_worker_get_address(m_worker, &addr, &addr_len);
    MEGRAY_ASSERT(status == UCS_OK, "failed to get ucp worker address");

    // allgather addr_len
    MEGRAY_CHECK(m_client->allgather(&addr_len, addr_lens, sizeof(size_t)));

    // find max addr_len
    size_t max_len = 0;
    for (size_t i = 0; i < m_nranks; i++) {
        if (addr_lens[i] > max_len) {
            max_len = addr_lens[i];
        }
    }

    // allgather addr
    char addrs[max_len * m_nranks];
    MEGRAY_CHECK(m_client->allgather(addr, addrs, max_len));
    ucp_worker_release_address(m_worker, addr);

    // set endpoint params
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;

    // create ucp endpoint
    m_eps.resize(m_nranks);
    for (size_t i = 0; i < m_nranks; i++) {
        if (i == m_rank)
            continue;
        ep_params.address =
                reinterpret_cast<const ucp_address_t*>(addrs + i * max_len);
        status = ucp_ep_create(m_worker, &ep_params, &m_eps[i]);
        MEGRAY_ASSERT(status == UCS_OK, "failed to create ucp endpoint");
    }

    return MEGRAY_OK;
}

Status UcxCommunicator::_send(const void* sendbuff, size_t size, uint32_t rank,
                              std::shared_ptr<Context> ctx) {
    // cuda stream synchronize
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // perform send recv
    char sync;
    MEGRAY_CHECK(_isend(sendbuff, size, rank));
    MEGRAY_CHECK(_irecv(&sync, sizeof(char), rank));  // synchronize
    MEGRAY_CHECK(_flush());
    return MEGRAY_OK;
}

Status UcxCommunicator::_recv(void* recvbuff, size_t size, uint32_t rank,
                              std::shared_ptr<Context> ctx) {
    // cuda stream synchronize
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // perform send recv
    char sync;
    MEGRAY_CHECK(_irecv(recvbuff, size, rank));
    MEGRAY_CHECK(_isend(&sync, sizeof(char), rank));  // synchronize
    MEGRAY_CHECK(_flush());
    return MEGRAY_OK;
}

Status UcxCommunicator::_isend(const void* sendbuff, size_t len,
                               uint32_t rank) {
    MEGRAY_ASSERT(rank != m_rank, "invalid send rank");
    std::lock_guard<std::mutex> lock(m_requests_mtx);
    // submit non-blocking send request to ucp
    void* ptr = ucp_tag_send_nb(m_eps[rank], sendbuff, len,
                                ucp_dt_make_contig(1), m_rank, send_cb);
    if (UCS_PTR_IS_PTR(ptr)) {
        m_requests.push_back(ptr);  // send request is pending
    } else if (UCS_PTR_STATUS(ptr) != UCS_OK) {
        return MEGRAY_UCX_ERR;
    }
    return MEGRAY_OK;
}

Status UcxCommunicator::_irecv(void* recvbuff, size_t len, uint32_t rank) {
    MEGRAY_ASSERT(rank != m_rank, "invalid recv rank");
    std::lock_guard<std::mutex> lock(m_requests_mtx);
    // submit non-blocking receive request to ucp
    // mask 0xffffffff means matching every bit of uint32
    void* ptr = ucp_tag_recv_nb(m_worker, recvbuff, len, ucp_dt_make_contig(1),
                                rank, 0xffffffff, recv_cb);
    if (UCS_PTR_IS_PTR(ptr)) {
        m_requests.push_back(ptr);
    } else if (UCS_PTR_STATUS(ptr) != UCS_OK) {  // receive request is pending
        return MEGRAY_UCX_ERR;
    }
    return MEGRAY_OK;
}

Status UcxCommunicator::_flush() {
    std::lock_guard<std::mutex> lock(m_requests_mtx);
    for (size_t i = 0; i < m_requests.size(); i++) {
        Request* req = (Request*)(m_requests[i]);
        // check flag "completed" of request handler
        while (req->completed == 0) {
            ucp_worker_progress(m_worker);
        }
        // release request handler
        req->completed = 0;
        ucp_request_release(req);
    }
    m_requests.clear();
    return MEGRAY_OK;
}

}  // namespace MegRay
