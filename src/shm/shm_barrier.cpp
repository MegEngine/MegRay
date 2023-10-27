#include "communicator.h"
namespace MegRay {

void ShmCommunicator::_shm_barrier_sum(volatile int* mutex) {
    int count{0};
    while (1) {
        count = 0;
        for (auto i = 0; i < m_nranks; i++) {
            count += mutex[i];
        }
        if (count == m_nranks)
            break;
        // TODO: declear the task spend how long
        // to save the cpu time
    }
};

void ShmCommunicator::_shm_barrier(volatile int* mutex) {
    if (m_rank == 0) {
        for (size_t i = 1;i < m_nranks;i++) {
            while(mutex[i]!=1);
        }
        mutex[0] = m_nranks;
        for (size_t i = 1;i < m_nranks;i++) {
            while(mutex[i]!=2);
        }
        mutex[0] = 0;
    } else {
        mutex[m_rank] = 1;
        while(mutex[0]!=m_nranks);
        mutex[m_rank] = 2;
        while(mutex[0]!=0);
        mutex[m_rank] = 0;
    }
};

}  // namespace MegRay
