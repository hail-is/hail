#include "hail/ObjectArray.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include "hail/Upcalls.h"
#include <jni.h>

namespace hail {

ObjectArray::ObjectArray(JNIEnv* env, jobjectArray objects) {
  ssize_t n = env->GetArrayLength(objects);
  vec_.resize(n);
  for (ssize_t j = 0; j < n; ++j) {
    jobject local_ref = env->GetObjectArrayElement(objects, j);
    vec_[j] = env->NewGlobalRef(local_ref);
    env->DeleteLocalRef(local_ref);
  }
}

ObjectArray::ObjectArray(JNIEnv* env, jobject a0) {
  vec_.resize(1);
  vec_[0] = env->NewGlobalRef(a0);
  env->DeleteLocalRef(a0);
}

ObjectArray::ObjectArray(JNIEnv* env, jobject a0, jobject a1) {
  vec_.resize(2);
  vec_[0] = env->NewGlobalRef(a0);
  env->DeleteLocalRef(a0);
  vec_[1] = env->NewGlobalRef(a1);
  env->DeleteLocalRef(a1);
}

ObjectArray::ObjectArray(JNIEnv* env, jobject a0, jobject a1, jobject a2) {
  vec_.resize(3);
  vec_[0] = env->NewGlobalRef(a0);
  env->DeleteLocalRef(a0);
  vec_[1] = env->NewGlobalRef(a1);
  env->DeleteLocalRef(a1);
  vec_[2] = env->NewGlobalRef(a2);
  env->DeleteLocalRef(a2);
}

ObjectArray::ObjectArray(JNIEnv* env, jobject a0, jobject a1, jobject a2, jobject a3) {
  vec_.resize(4);
  vec_[0] = env->NewGlobalRef(a0);
  env->DeleteLocalRef(a0);
  vec_[1] = env->NewGlobalRef(a1);
  env->DeleteLocalRef(a1);
  vec_[2] = env->NewGlobalRef(a2);
  env->DeleteLocalRef(a2);
  vec_[3] = env->NewGlobalRef(a3);
  env->DeleteLocalRef(a3);
}

ObjectArray::~ObjectArray() {
  // Not necessarily the same env as used in constructor
  UpcallEnv up;
  for (ssize_t j = vec_.size(); --j >= 0;) {
    up.env()->DeleteGlobalRef(vec_[j]);
  }
}

// Constructor methods accessible from is.hail.nativecode.ObjectArray

NATIVEMETHOD(void, ObjectArray, nativeCtorArray)(
  JNIEnv* env,
  jobject thisJ,
  jobjectArray arrayJ
) {
  NativeObjPtr ptr = std::make_shared<ObjectArray>(env, arrayJ);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, ObjectArray, nativeCtorO1)(
  JNIEnv* env,
  jobject thisJ,
  jobject a0
) {
  NativeObjPtr ptr = std::make_shared<ObjectArray>(env, a0);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, ObjectArray, nativeCtorO2)(
  JNIEnv* env,
  jobject thisJ,
  jobject a0,
  jobject a1
) {
  NativeObjPtr ptr = std::make_shared<ObjectArray>(env, a0, a1);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, ObjectArray, nativeCtorO3)(
  JNIEnv* env,
  jobject thisJ,
  jobject a0,
  jobject a1,
  jobject a2
) {
  NativeObjPtr ptr = std::make_shared<ObjectArray>(env, a0, a1, a2);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, ObjectArray, nativeCtorO4)(
  JNIEnv* env,
  jobject thisJ,
  jobject a0,
  jobject a1,
  jobject a2,
  jobject a3
) {
  NativeObjPtr ptr = std::make_shared<ObjectArray>(env, a0, a1, a2, a3);
  init_NativePtr(env, thisJ, &ptr);
}

} // end hail