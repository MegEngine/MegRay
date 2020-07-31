/**
 * \file src/communicator.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megray/core/communicator.h"

namespace MegRay {

Status Communicator::init(const char* master_ip, int port) {
    m_client = std::make_shared<Client>(m_nranks, m_rank);
    MEGRAY_CHECK(m_client->connect(master_ip, port));
    return do_init();
}

} // namespace MegRay
