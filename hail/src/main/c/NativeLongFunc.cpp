#include "hail/NativeMethod.h"
#include "hail/NativePtr.h"
#include "hail/NativeStatus.h"
#include <cassert>
#include <jni.h>

namespace hail {

namespace {

using NativeFunc = NativeFuncObj<int64_t>;

NativeFunc* to_NativeFunc(JNIEnv* env, jobject thisJ) {
  // Should be a dynamic_cast, but RTTI causes trouble
  return static_cast<NativeFunc*>(get_from_NativePtr(env, thisJ));
}

} // end anon

NATIVEMETHOD(jlong, NativeLongFuncL0, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status);
}

NATIVEMETHOD(jlong, NativeLongFuncL1, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0);
}

NATIVEMETHOD(jlong, NativeLongFuncL2, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1);
}

NATIVEMETHOD(jlong, NativeLongFuncL3, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2
  
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1, a2);
}

NATIVEMETHOD(jlong, NativeLongFuncL4, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1, a2, a3);
}

NATIVEMETHOD(jlong, NativeLongFuncL5, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1, a2, a3, a4);
}

NATIVEMETHOD(jlong, NativeLongFuncL6, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1, a2, a3, a4, a5);
}

NATIVEMETHOD(jlong, NativeLongFuncL7, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1, a2, a3, a4, a5, a6);
}

NATIVEMETHOD(jlong, NativeLongFuncL8, nativeApply)(
  JNIEnv* env,
  jobject thisJ,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6,
  jlong a7
) {
  auto f = to_NativeFunc(env, thisJ);
  assert(f);
  auto status = reinterpret_cast<NativeStatus*>(st);
  return f->func_(status, a0, a1, a2, a3, a4, a5, a6, a7);
}

} // end hail
