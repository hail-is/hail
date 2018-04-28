//
// src/main/c/NativeFunc.cpp - native funcs for Scala NativeLongFunc
//
// Richard Cownie, Hail Team, 2018-04-18
//
#include "hail/NativePtr.h"
#include "hail/CommonDefs.h"
#include <assert.h>
#include <jni.h>

NAMESPACE_BEGIN(hail)

NAMESPACE_BEGIN_ANON

typedef NativeFuncObj<long> NativeFunc;

NativeFunc* toNativeFunc(JNIEnv* env, jobject thisObj) {
  // Should be a dynamic_cast, but RTTI causes trouble
  return reinterpret_cast<NativeFunc*>(getFromNativePtr(env, thisObj));
}

NAMESPACE_END_ANON

NATIVEMETHOD(long, NativeLongFuncL0, apply)(
  JNIEnv* env,
  jobject thisObj
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_();
}

NATIVEMETHOD(long, NativeLongFuncL1, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0);
}

NATIVEMETHOD(long, NativeLongFuncL2, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1);
}

NATIVEMETHOD(long, NativeLongFuncL3, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1,
  jlong a2
  
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1, a2);
}

NATIVEMETHOD(long, NativeLongFuncL4, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1, a2, a3);
}

NATIVEMETHOD(long, NativeLongFuncL5, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1, a2, a3, a4);
}

NATIVEMETHOD(long, NativeLongFuncL6, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1, a2, a3, a4, a5);
}

NATIVEMETHOD(long, NativeLongFuncL7, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1, a2, a3, a4, a5, a6);
}

NATIVEMETHOD(long, NativeLongFuncL8, apply)(
  JNIEnv* env,
  jobject thisObj,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6,
  jlong a7
) {
  auto funcObj = toNativeFunc(env, thisObj);
  assert(funcObj);
  return funcObj->func_(a0, a1, a2, a3, a4, a5, a6, a7);
}

NAMESPACE_END(hail)
