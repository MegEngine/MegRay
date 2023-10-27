#include "megray/debug.h"

#include <cstdarg>

namespace MegRay {

void MEGRAY_LOG(const char* level, const char* file, int line, const char* fmt,
                ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[%s]\t%s:%d, ", level, file, line);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

}  // namespace MegRay
