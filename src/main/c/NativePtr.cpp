#include "hail/NativePtr.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <assert.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

namespace hail {

// JNI interface functions corresponding to Scala NativeBase and NativePtr

#define NATIVEFUNC(typ) \
  extern "C" JNIEXPORT typ JNICALL

// According to the C++ standard, pointers to different types are assuming to be
// non-aliased, unless they go through a "char*" which can be an alias for any type
#define ALIAS_AS_LONGVEC(p) \
  (reinterpret_cast<long*>(reinterpret_cast<char*>(p)))

#define ALIAS_AS_NATIVEOBJPTR(p) \
  (reinterpret_cast<NativeObjPtr*>(reinterpret_cast<char*>(p)))

namespace {

bool is_info_ready = false;
jfieldID addrA_id;
jfieldID addrB_id;

void init_info(JNIEnv* env) {
  auto cl = env->FindClass("is/hail/nativecode/NativeBase");
  addrA_id = env->GetFieldID(cl, "addrA", "J");
  addrB_id = env->GetFieldID(cl, "addrB", "J");
  is_info_ready = true;
}

} // end anon

NativeObj* get_from_NativePtr(JNIEnv* env, jobject obj) {
  if (!is_info_ready) init_info(env);
  long addrA = env->GetLongField(obj, addrA_id);
  return reinterpret_cast<NativeObj*>(addrA);
}

void init_NativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  if (!is_info_ready) init_info(env);
  // Ignore previous values in NativePtr
  long* vec = ALIAS_AS_LONGVEC(ptr);
  env->SetLongField(obj, addrA_id, vec[0]);
  env->SetLongField(obj, addrB_id, vec[1]);
  // And clear the ptr without changing refcount
  vec[0] = 0;
  vec[1] = 0;
}

void move_to_NativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  if (!is_info_ready) init_info(env);
  long addrA = env->GetLongField(obj, addrA_id);
  if (addrA) {
    // We need to reset() the existing NativePtr
    long oldVec[2];
    oldVec[0] = addrA;
    oldVec[1] = env->GetLongField(obj, addrB_id);
    auto oldPtr = ALIAS_AS_NATIVEOBJPTR(&oldVec[0]);
    oldPtr->reset();
  }
  long* vec = ALIAS_AS_LONGVEC(ptr);
  env->SetLongField(obj, addrA_id, vec[0]);
  env->SetLongField(obj, addrB_id, vec[1]);
  // And clear the ptr without changing refcount
  vec[0] = 0;
  vec[1] = 0;
}

NATIVEMETHOD(void, NativeBase, nativeCopyCtor)(
  JNIEnv* env,
  jobject thisJ,
  jlong b_addrA,
  jlong b_addrB
) {
  if (!is_info_ready) init_info(env);
  auto obj = reinterpret_cast<NativeObj*>(b_addrA);
  // This adds a new reference to the object
  auto ptr = (obj ? obj->shared_from_this() : NativeObjPtr());
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativeBase, copyAssign)(
  JNIEnv* env,
  jobject thisJ,
  jobject srcJ
) {
  if (thisJ == srcJ) return;
  long vecA[2];
  vecA[0] = env->GetLongField(thisJ, addrA_id);
  vecA[1] = env->GetLongField(thisJ, addrB_id);
  long vecB[2];
  vecB[0] = env->GetLongField(srcJ, addrA_id);
  vecB[1] = env->GetLongField(srcJ, addrB_id);
  auto ptrA = ALIAS_AS_NATIVEOBJPTR(&vecA[0]);
  auto ptrB = ALIAS_AS_NATIVEOBJPTR(&vecB[0]);
  if (*ptrA != *ptrB) {
    *ptrA = *ptrB;
    env->SetLongField(thisJ, addrA_id, vecA[0]);
    env->SetLongField(thisJ, addrB_id, vecA[1]);
  }
}

NATIVEMETHOD(void, NativeBase, moveAssign)(
  JNIEnv* env,
  jobject thisJ,
  jobject srcJ
) {
  if (thisJ == srcJ) return;
  long vecA[2];
  vecA[0] = env->GetLongField(thisJ, addrA_id);
  vecA[1] = env->GetLongField(thisJ, addrB_id);
  long vecB[2];
  vecB[0] = env->GetLongField(srcJ, addrA_id);
  vecB[1] = env->GetLongField(srcJ, addrB_id);
  auto ptrA = ALIAS_AS_NATIVEOBJPTR(&vecA[0]);
  auto ptrB = ALIAS_AS_NATIVEOBJPTR(&vecB[0]);
  *ptrA = std::move(*ptrB);
  env->SetLongField(thisJ, addrA_id, vecA[0]);
  env->SetLongField(thisJ, addrB_id, vecA[1]);
  env->SetLongField(srcJ, addrA_id, vecB[0]);
  env->SetLongField(srcJ, addrB_id, vecB[1]);
}

NATIVEMETHOD(void, NativeBase, nativeReset)(
  JNIEnv* env,
  jobject thisJ,
  jlong addrA,
  jlong addrB
) {
  long vec[2];
  vec[0] = addrA;
  vec[1] = addrB;
  auto ptr = ALIAS_AS_NATIVEOBJPTR(&vec[0]);
  ptr->reset();
  // The Scala object fields are cleared in the wrapper
}

NATIVEMETHOD(long, NativeBase, nativeUseCount)(
  JNIEnv* env,
  jobject thisJ,
  jlong addrA,
  jlong addrB
) {
  long vec[2];
  vec[0] = addrA;
  vec[1] = addrB;
  auto ptr = ALIAS_AS_NATIVEOBJPTR(&vec[0]);
  return ptr->use_count();
}

// We have constructors corresponding to std::make_shared<T>(...)
// with various numbers of long arguments.

NATIVEMETHOD(void, NativePtr, nativePtrFuncL0)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)();
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL1)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL2)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL3)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1, a2);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL4)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1, a2, a3);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL5)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL6)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4, a5);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL7)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4, a5, a6);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL8)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6,
  jlong a7
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4, a5, a6, a7);
  init_NativePtr(env, thisJ, &ptr);
}

// Test code for nativePtrFunc

class TestObjA : public hail::NativeObj {
private:
  std::vector<long> vec_;
  
public:
  TestObjA() : vec_() {
    fprintf(stderr, "DEBUG: TestObjA::ctor()\n");
  }
  
  TestObjA(long a0) : vec_({a0}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx)\n", a0);
  }

  TestObjA(long a0, long a1) : vec_({a0, a1}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx, %08lx)\n", a0, a1);
  }

  TestObjA(long a0, long a1, long a2) : vec_({a0,a1,a2}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx, %08lx, %08lx)\n", a0, a1, a2);
  }

  TestObjA(long a0, long a1, long a2, long a3) : vec_({a0,a1,a2,a3}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx, %08lx, %08lx, %08lx)\n",
      a0, a1, a2, a3);
  }

  virtual ~TestObjA() {
    fprintf(stderr, "DEBUG: TestObjA::dtor() size %lu\n", vec_.size());
  }
};

NativeObjPtr nativePtrFuncTestObjA0() {
  return MAKE_NATIVE(TestObjA);
}

NativeObjPtr nativePtrFuncTestObjA1(long a0) {
  return MAKE_NATIVE(TestObjA, a0);
}

NativeObjPtr nativePtrFuncTestObjA2(long a0, long a1) {
  return MAKE_NATIVE(TestObjA, a0, a1);
}

NativeObjPtr nativePtrFuncTestObjA3(long a0, long a1, long a2) {
  return MAKE_NATIVE(TestObjA, a0, a1, a2);
}

NativeObjPtr nativePtrFuncTestObjA4(NativeObjPtr* out, long a0, long a1, long a2, long a3) {
  return MAKE_NATIVE(TestObjA, a0, a1, a2, a3);
}

} // end hail

