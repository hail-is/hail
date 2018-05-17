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

namespace {

class AlignedBuf {
public:
  char buf_[2*sizeof(long)];
  long force_align_;

public:
  AlignedBuf() {
    set_addrA(0);
    set_addrB(0);
  }

  inline NativeObjPtr& as_NativeObjPtr() {
    return *reinterpret_cast<NativeObjPtr*>(&buf_[0]);
  }
  
  long get_addrA() const;
  long get_addrB() const;
  void set_addrA(long v);
  void set_addrB(long v);
};

// We use non-inline methods to try to defeat over-aggressive reordering
// of aliased type-punned accesses.  The cost is probably small relative
// to the overhead of the Scala-to-C++ JNI call.

long AlignedBuf::get_addrA() const { return *(long*)(&buf_[0]); }  
long AlignedBuf::get_addrB() const { return *(long*)(&buf_[8]); }
void AlignedBuf::set_addrA(long v) { *(long*)&buf_[0] = v; }
void AlignedBuf::set_addrB(long v) { *(long*)&buf_[8] = v; }

bool is_info_ready = false;
jfieldID addrA_id;
jfieldID addrB_id;

void init_info(JNIEnv* env) {
  auto cl = env->FindClass("is/hail/nativecode/NativeBase");
  addrA_id = env->GetFieldID(cl, "addrA", "J");
  addrB_id = env->GetFieldID(cl, "addrB", "J");
  is_info_ready = true;
  // Check that std::shared_ptr matches our assumptions
  auto ptr = std::make_shared<NativeObj>();
  AlignedBuf* buf = reinterpret_cast<AlignedBuf*>(&ptr);
  assert(sizeof(ptr) == 2*sizeof(long));
  assert(buf->get_addrA() == reinterpret_cast<long>(ptr.get()));
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
  AlignedBuf buf;
  buf.as_NativeObjPtr() = std::move(*ptr);
  env->SetLongField(obj, addrA_id, buf.get_addrA());
  env->SetLongField(obj, addrB_id, buf.get_addrB());
}

void move_to_NativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  if (!is_info_ready) init_info(env);
  long addrA = env->GetLongField(obj, addrA_id);
  if (addrA) {
    // We need to reset() the existing NativePtr
    AlignedBuf old;
    old.set_addrA(addrA);
    old.set_addrB(env->GetLongField(obj, addrB_id));
    old.as_NativeObjPtr().reset();
  }
  AlignedBuf buf;
  buf.as_NativeObjPtr() = std::move(*ptr);
  env->SetLongField(obj, addrA_id, buf.get_addrA());
  env->SetLongField(obj, addrB_id, buf.get_addrB());
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
  AlignedBuf bufA;
  bufA.set_addrA(env->GetLongField(thisJ, addrA_id));
  bufA.set_addrB(env->GetLongField(thisJ, addrB_id));
  AlignedBuf bufB;
  bufB.set_addrA(env->GetLongField(srcJ, addrA_id));
  bufB.set_addrB(env->GetLongField(srcJ, addrB_id));
  auto& ptrA = bufA.as_NativeObjPtr();
  auto& ptrB = bufB.as_NativeObjPtr();
  if (ptrA != ptrB) {
    // This will change the contents of vecA
    ptrA = ptrB;
    env->SetLongField(thisJ, addrA_id, bufA.get_addrA());
    env->SetLongField(thisJ, addrB_id, bufA.get_addrB());
  }
}

NATIVEMETHOD(void, NativeBase, moveAssign)(
  JNIEnv* env,
  jobject thisJ,
  jobject srcJ
) {
  if (thisJ == srcJ) return;
  AlignedBuf bufA;
  bufA.set_addrA(env->GetLongField(thisJ, addrA_id));
  bufA.set_addrB(env->GetLongField(thisJ, addrB_id));
  AlignedBuf bufB;
  bufB.set_addrA(env->GetLongField(srcJ, addrA_id));
  bufB.set_addrB(env->GetLongField(srcJ, addrB_id));
  bufA.as_NativeObjPtr() = std::move(bufB.as_NativeObjPtr());
  env->SetLongField(thisJ, addrA_id, bufA.get_addrA());
  env->SetLongField(thisJ, addrB_id, bufA.get_addrB());
  env->SetLongField(srcJ, addrA_id, bufB.get_addrA());
  env->SetLongField(srcJ, addrB_id, bufB.get_addrB());
}

NATIVEMETHOD(void, NativeBase, nativeReset)(
  JNIEnv* env,
  jobject thisJ,
  jlong addrA,
  jlong addrB
) {
  AlignedBuf bufA;
  bufA.set_addrA(addrA);
  bufA.set_addrB(addrB);
  bufA.as_NativeObjPtr().reset();
  // The Scala object fields are cleared in the wrapper
}

NATIVEMETHOD(long, NativeBase, nativeUseCount)(
  JNIEnv* env,
  jobject thisJ,
  jlong addrA,
  jlong addrB
) {
  AlignedBuf bufA;
  bufA.set_addrA(addrA);
  bufA.set_addrB(addrB);
  return bufA.as_NativeObjPtr().use_count();
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

