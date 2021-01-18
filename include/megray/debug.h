/**
 * \file include/megray/debug.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

namespace MegRay {

typedef enum { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 } LogLevel;

#ifndef MEGRAY_LOG_LEVEL
#define MEGRAY_LOG_LEVEL 2
#endif

void MEGRAY_LOG(const char* level, const char* file, int line, const char* fmt,
                ...);

#define MEGRAY_DEBUG(fmt...)                                      \
    do {                                                          \
        if (MegRay::LogLevel::DEBUG >= MEGRAY_LOG_LEVEL) {        \
            MegRay::MEGRAY_LOG("DEBUG", __FILE__, __LINE__, fmt); \
        }                                                         \
    } while (0)

#define MEGRAY_INFO(fmt...)                                      \
    do {                                                         \
        if (MegRay::LogLevel::INFO >= MEGRAY_LOG_LEVEL) {        \
            MegRay::MEGRAY_LOG("INFO", __FILE__, __LINE__, fmt); \
        }                                                        \
    } while (0)

#define MEGRAY_WARN(fmt...)                                      \
    do {                                                         \
        if (MegRay::LogLevel::WARN >= MEGRAY_LOG_LEVEL) {        \
            MegRay::MEGRAY_LOG("WARN", __FILE__, __LINE__, fmt); \
        }                                                        \
    } while (0)

#define MEGRAY_ERROR(fmt...)                                      \
    do {                                                          \
        if (MegRay::LogLevel::ERROR >= MEGRAY_LOG_LEVEL) {        \
            MegRay::MEGRAY_LOG("ERROR", __FILE__, __LINE__, fmt); \
        }                                                         \
    } while (0)

class Exception : public std::runtime_error {
public:
    Exception() = default;
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

#define MEGRAY_THROW(message) throw MegRay::Exception(message)

#define MEGRAY_ASSERT(expr, fmt...)                      \
    do {                                                 \
        if (!(expr)) {                                   \
            MEGRAY_ERROR("assertion failed: %s", #expr); \
            MEGRAY_ERROR(fmt);                           \
            MEGRAY_THROW("assertion failed");            \
        }                                                \
    } while (0)

}  // namespace MegRay
