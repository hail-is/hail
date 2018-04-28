//
// src/main/c/NativeStatus.cpp - status/error-reporting for native C++ funcs
//
// Richard Cownie, Hail Team, 2018-04-24
//
#include "hail/NativeStatus.h"
#include "hail/CommonDefs.h"
#include "hail/NativePtr.h"
#include <jni.h>

NAMESPACE_BEGIN(hail)

NATIVEMETHOD(long, NativeStatus, nativeCtorErrnoOffset)(
  JNIEnv* env,
  jobject thisObj
) {
  auto status = std::make_shared<NativeStatus>();
  long errnoOffset = ((long)&status->errno_) - (long)status.get();
  auto ptr = std::static_pointer_cast<NativeObj>(status);
  initNativePtr(env, thisObj, &ptr);
  return(errnoOffset);
}

NATIVEMETHOD(jstring, NativeStatus, getMsg)(
  JNIEnv* env,
  jobject thisObj
) {
  auto status = reinterpret_cast<NativeStatus*>(getFromNativePtr(env, thisObj));
  const char* s = ((status->errno_ == 0) ? "NoError" : status->msg_.c_str());
  return env->NewStringUTF(s);
}

NATIVEMETHOD(jstring, NativeStatus, getLocation)(
  JNIEnv* env,
  jobject thisObj
) {
  auto status = reinterpret_cast<NativeStatus*>(getFromNativePtr(env, thisObj));
  const char* s = ((status->errno_ == 0) ? "NoLocation" : status->location_.c_str());
  return env->NewStringUTF(s);
}

NAMESPACE_END(hail)
