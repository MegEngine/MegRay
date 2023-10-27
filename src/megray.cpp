#include "megray.h"

#ifdef MEGRAY_WITH_NCCL
#include "nccl/communicator.h"
#endif

#ifdef MEGRAY_WITH_UCX
#include "ucx/communicator.h"
#endif

#ifdef MEGRAY_WITH_RCCL
#include "rccl/communicator.h"
#endif

#ifdef MEGRAY_WITH_SHM
#include "shm/communicator.h"
#endif

#ifdef MEGRAY_WITH_CNCL
#include "cncl/communicator.h"
#endif

namespace MegRay {

std::shared_ptr<Communicator> get_communicator(uint32_t nranks, uint32_t rank,
                                               Backend backend) {
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
#ifdef MEGRAY_WITH_RCCL
        case MEGRAY_RCCL:
            comm = std::make_shared<RcclCommunicator>(nranks, rank);
            break;
#endif
#ifdef MEGRAY_WITH_SHM
        case MEGRAY_SHM:
            comm = std::make_shared<ShmCommunicator>(nranks, rank);
            break;
#endif
#ifdef MEGRAY_WITH_CNCL
        case MEGRAY_CNCL:
            comm = std::make_shared<CnclCommunicator>(nranks, rank);
            break;
#endif
        default:
            MEGRAY_THROW("unknown backend");
    }
    return comm;
}

}  // namespace MegRay
