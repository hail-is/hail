#ifndef HAIL_NATIVEOBJ_H
#define HAIL_NATIVEOBJ_H 1

#include <memory>

namespace hail {

// The lifetime of off-heap objects may be controlled by any combination
// of C++ std::shared_ptr<T> and Jvm-side NativePtr objects.  But classes
// used in this way must inherit from NativeObj.

class NativeObj :
  public std::enable_shared_from_this<NativeObj> {
  public:
    // Objects managed with a Jvm-side NativePtr need to be destroyed
    // without knowing their precise type, so they must have a virtual
    // destructor.
    virtual ~NativeObj() { }

    // Subclasses should override getClassName, which may be helpful
    // for debugging.
    virtual const char* get_class_name() { return "NativeObj"; }
};

// On the Jvm side, we don't have distinct classes/types for NativePtr
// to different off-heap objects - we treat them all as NativeObjPtr.
using NativeObjPtr = std::shared_ptr<NativeObj>;

} // end hail

#endif
