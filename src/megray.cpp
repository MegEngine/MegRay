#include "megray/megray.h"

#ifdef MEGRAY_WITH_NCCL
#include "megray/nccl/communicator.h"
#endif

#ifdef MEGRAY_WITH_UCX
#include "megray/ucx/communicator.h"
#endif

namespace MegRay{

std::shared_ptr<Communicator> get_communicator(uint32_t nranks, uint32_t rank, Backend backend) {
    std::shared_ptr<Communicator> comm;
    switch (backend) {
        #ifdef MEGRAY_WITH_NCCL
        case MEGRAY_NCCL:
            comm = std::make_shared<NcclCommunicator>(nranks, rank);
            break;
        #endif
        #ifdef MEGRAY_WITH_UCX
        case MEGRAY_UCX:
            comm = std::make_shared<UcxCommunicator>(nranks, rank);
            break;
        #endif
        default:
            MEGRAY_THROW("unknown backend");
    }
    return comm;
}

}