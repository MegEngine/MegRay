/**
 * \file src/server.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <mutex>

#include "megray/core/common.h"

namespace MegRay {

char* get_host_ip();

int get_free_port();

// create megray server
Status create_server(uint32_t nranks, int port);

} // namespace MegRay
