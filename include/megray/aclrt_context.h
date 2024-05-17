#pragma once
#include <memory>

#include "megray/common.h"
#include "megray/context.h"

#ifdef MEGRAY_WITH_HCCL

#include <acl/acl.h>

namespace MegRay {

class AclrtContext : public Context {
public:
    AclrtContext(aclrtStream stream) : m_stream{stream} {}
    static std::shared_ptr<AclrtContext> make(aclrtStream stream) {
        return std::make_shared<AclrtContext>(stream);
    }
    ContextType type() const override { return MEGRAY_CTX_ACLRT; }
    aclrtStream get_stream() { return m_stream; }

private:
    aclrtStream m_stream;
};

}  // namespace MegRay

#endif
