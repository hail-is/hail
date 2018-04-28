#include "hail/NativePtr.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <assert.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

using hail::NativeObjPtr;
using hail::NativeObj;

NAMESPACE_BEGIN(hail)

//
// JNI interface functions corresponding to Scala NativeBase and NativePtr
//

#define NATIVEFUNC(typ) \
  extern "C" JNIEXPORT typ JNICALL

#define volatile

namespace { // anonymous

typedef void MakeNativeFunc(NativeObjPtr*, ...);

bool isInfoReady = false;
jfieldID addrAFieldID;
jfieldID addrBFieldID;

void initInfo(JNIEnv* env) {
  auto classID = env->FindClass("is/hail/nativecode/NativeBase");
  addrAFieldID = env->GetFieldID(classID, "addrA", "J");
  addrBFieldID = env->GetFieldID(classID, "addrB", "J");
  isInfoReady = true;
}

} // end anonymous

NativeObj* getFromNativePtr(JNIEnv* env, jobject obj) {
  if (!isInfoReady) initInfo(env);
  long addrA = env->GetLongField(obj, addrAFieldID);
  return reinterpret_cast<NativeObj*>(addrA);
}

void initNativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  if (!isInfoReady) initInfo(env);
  // Ignore previous values in NativePtr
  long* vec = reinterpret_cast<long*>(ptr);
  env->SetLongField(obj, addrAFieldID, vec[0]);
  env->SetLongField(obj, addrBFieldID, vec[1]);
  // And clear the ptr without changing refcount
  vec[0] = 0;
  vec[1] = 0;
}

void moveToNativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  if (!isInfoReady) initInfo(env);
  long addrA = env->GetLongField(obj, addrAFieldID);
  if (addrA) {
    // We need to reset() the existing NativePtr
    long oldVec[2];
    oldVec[0] = addrA;
    oldVec[1] = env->GetLongField(obj, addrBFieldID);
    auto oldPtr = reinterpret_cast<NativeObjPtr*>(&oldVec[0]);
    oldPtr->reset();
  }
  long* vec = reinterpret_cast<long*>(ptr);
  env->SetLongField(obj, addrAFieldID, vec[0]);
  env->SetLongField(obj, addrBFieldID, vec[1]);
  // And clear the ptr without changing refcount
  vec[0] = 0;
  vec[1] = 0;
}

NATIVEMETHOD(void, NativeBase, nativeCopyCtor)(
  JNIEnv* env,
  jobject thisObj,
  jlong b_addrA,
  jlong b_addrB
) {
  if (!isInfoReady) initInfo(env);
  auto obj = reinterpret_cast<NativeObj*>(b_addrA);
  // This adds a new reference to the object
  auto ptr = (obj ? obj->shared_from_this() : NativeObjPtr());
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativeBase, copyAssign)(
  JNIEnv* env,
  jobject thisObj,
  jobject srcObj
) {
  if (thisObj == srcObj) return;
  volatile long vecA[2];
  vecA[0] = env->GetLongField(thisObj, addrAFieldID);
  vecA[1] = env->GetLongField(thisObj, addrBFieldID);
  volatile long vecB[2];
  vecB[0] = env->GetLongField(srcObj, addrAFieldID);
  vecB[1] = env->GetLongField(srcObj, addrBFieldID);
  auto ptrA = reinterpret_cast<NativeObjPtr*>(&vecA[0]);
  auto ptrB = reinterpret_cast<NativeObjPtr*>(&vecB[0]);
 if (*ptrA != *ptrB) {
    *ptrA = *ptrB;
    env->SetLongField(thisObj, addrAFieldID, vecA[0]);
    env->SetLongField(thisObj, addrBFieldID, vecA[1]);
  }
}

NATIVEMETHOD(void, NativeBase, moveAssign)(
  JNIEnv* env,
  jobject thisObj,
  jobject srcObj
) {
  if (thisObj == srcObj) return;
  volatile long vecA[2];
  vecA[0] = env->GetLongField(thisObj, addrAFieldID);
  vecA[1] = env->GetLongField(thisObj, addrBFieldID);
  volatile long vecB[2];
  vecB[0] = env->GetLongField(srcObj, addrAFieldID);
  vecB[1] = env->GetLongField(srcObj, addrBFieldID);
  auto ptrA = reinterpret_cast<NativeObjPtr*>(&vecA[0]);
  auto ptrB = reinterpret_cast<NativeObjPtr*>(&vecB[0]);
  *ptrA = std::move(*ptrB);
  env->SetLongField(thisObj, addrAFieldID, vecA[0]);
  env->SetLongField(thisObj, addrBFieldID, vecA[1]);
  env->SetLongField(srcObj, addrAFieldID, vecB[0]);
  env->SetLongField(srcObj, addrBFieldID, vecB[1]);
  
}

NATIVEMETHOD(void, NativeBase, nativeReset)(
  JNIEnv* env,
  jobject thisObj,
  jlong addrA,
  jlong addrB
) {
  volatile long vec[2];
  vec[0] = addrA;
  vec[1] = addrB;
  auto ptr = reinterpret_cast<NativeObjPtr*>(&vec[0]);
  ptr->reset();
  // The Scala object fields are cleared in the wrapper
}

NATIVEMETHOD(long, NativeBase, nativeUseCount)(
  JNIEnv* env,
  jobject thisObj,
  jlong addrA,
  jlong addrB
) {
  volatile long vec[2];
  vec[0] = addrA;
  vec[1] = addrB;
  auto ptr = reinterpret_cast<NativeObjPtr*>(&vec[0]);
  return ptr->use_count();
}

//
// We have constructors corresponding to std::make_shared<T>(...)
// with various numbers of long arguments.
//

NATIVEMETHOD(void, NativePtr, nativePtrFuncL0)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)();
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL1)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0
) {
  FILE* f = fopen("/tmp/a.log", "w");
  fprintf(f, "nativePtrFuncL1(%ld) ...\n", a0); fflush(f);
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0);
  NativeObjPtr ptr2 = ptr;
  fprintf(f, "ptr.get %p, use_count %lu\n", ptr.get(), ptr.use_count()); fflush(f);
  initNativePtr(env, thisObj, &ptr);
  fprintf(f, "ptr.get %p\n", ptr.get()); fflush(f);
  fclose(f);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL2)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0,
  jlong a1
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL3)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1, a2);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL4)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1, a2, a3);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL5)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL6)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4, a5);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL7)(
  JNIEnv* env,
  jobject thisObj,
  jlong funcObjAddr,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6
) {
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4, a5, a6);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL8)(
  JNIEnv* env,
  jobject thisObj,
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
  NativeObjPtr ptr = getPtrFuncN(funcObjAddr)(a0, a1, a2, a3, a4, a5, a6, a7);
  initNativePtr(env, thisObj, &ptr);
}

//
// Test code for nativePtrFunc
//

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

//
// Note that std::static_pointer_cast will only work when the type of
// the object is a subclass of NativeObj, which is what we need.
//
#define CastNativeObjPtr(ptr) \
  std::static_pointer_cast<hail::NativeObj>(ptr)

NativeObjPtr nativePtrFuncTestObjA0() {
  return CastNativeObjPtr(std::make_shared<TestObjA>());
}

NativeObjPtr nativePtrFuncTestObjA1(long a0) {
  return CastNativeObjPtr(std::make_shared<TestObjA>(a0));
}

NativeObjPtr nativePtrFuncTestObjA2(long a0, long a1) {
  return CastNativeObjPtr(std::make_shared<TestObjA>(a0, a1));
}

NativeObjPtr nativePtrFuncTestObjA3(long a0, long a1, long a2) {
  return CastNativeObjPtr(std::make_shared<TestObjA>(a0, a1, a2));
}

NativeObjPtr nativePtrFuncTestObjA4(NativeObjPtr* out, long a0, long a1, long a2, long a3) {
  return CastNativeObjPtr(std::make_shared<TestObjA>(a0, a1, a2, a3));
}

NAMESPACE_END(hail)


