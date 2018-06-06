#include "hail/Region.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include <jni.h>

namespace hail {

#define REGIONMETHOD(rtype, scala_class, scala_method) \
  extern "C" __attribute__((visibility("default"))) \
    rtype Java_is_hail_annotations_##scala_class##_##scala_method

REGIONMETHOD(void, Region, nativeCtor)(
  JNIEnv* env,
  jobject thisJ
) {
  NativeObjPtr ptr = std::make_shared<Region>();
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, clearButKeepMem)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  r->clear_but_keep_mem();
}

REGIONMETHOD(void, Region, nativeAlign)(
  JNIEnv* env,
  jobject thisJ,
  jlong a
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  r->align(a);
}

REGIONMETHOD(long, Region, nativeAlignAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong a,
  jlong n
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<long>(r->allocate(a, n));
}

REGIONMETHOD(long, Region, nativeAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong n
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<long>(r->allocate(n));
}

} // end hail
