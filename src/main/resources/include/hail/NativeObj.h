#ifndef HAIL_NATIVEOBJ_H
#define HAIL_NATIVEOBJ_H 1
//
// Declarations needed for C++ code to be integrated with Hail 
//
// Richard Cownie, Hail Team, 2018-04-03
//
#include "hail/CommonDefs.h"
#include <memory>

NAMESPACE_BEGIN(hail)

//
// The lifetime of off-heap objects may be controlled by any combination
// of C++ std::shared_ptr<T> and Jvm-side NativePtr objects.  But classes
// used in this way must inherit from NativeObj.
//

class NativeObj :
  public std::enable_shared_from_this<NativeObj> {
  public:
    //
    // Objects managed with a Jvm-side NativePtr need to be destroyed
    // without knowing their precise type, so they must have a virtual
    // destructor.
    //
    virtual ~NativeObj() { }
    
    //
    // Subclasses should override getClassName, which may be helpful
    // for debugging.
    //
    virtual const char* getClassName() { return "NativeObj"; }
    
    //
    // If a subclass wants to make some fields visible to Jvm Unsafe
    // access, it can publish their offsets.  This is probably only
    // useful for integer types, so we pass in one parameter to
    // identify the fieldSize as 1, 2, 4, or 8 bytes.
    //
    virtual long getFieldOffset(int fieldSize, const char* fieldName) { return(-1); }
};

//
// On the Jvm side, we don't have distinct classes/types for NativePtr
// to different off-heap objects - we treat them all as NativeObjPtr.
//
typedef std::shared_ptr<NativeObj> NativeObjPtr;

#define MAKE_NATIVE(ObjT, ...) \
  (std::static_pointer_cast<NativeObj>(std::make_shared<ObjT>(__VA_ARGS__)))
//
// Upcast conversion of pointer-to-subclass to pointer-to-base
//

template<typename T>
inline NativeObjPtr toNativeObjPtr(const std::shared_ptr<T>& a) {
  return std::static_pointer_cast<NativeObj>(a);
};

NAMESPACE_END(hail)

#endif
