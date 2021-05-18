/**
 * \file src/shm/shm_barrier.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

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
