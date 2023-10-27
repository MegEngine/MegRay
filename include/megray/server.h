#pragma once

#include "megray/common.h"

namespace MegRay {

char* get_host_ip();

int get_free_port();

// create megray server
Status create_server(uint32_t nranks, int port);

}  // namespace MegRay
