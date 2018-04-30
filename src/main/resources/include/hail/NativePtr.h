#ifndef HAIL_NATIVEPTR_H
#define HAIL_NATIVEPTR_H 1
//
// include/hail/NativePtr.h
//
// Richard Cownie, Hail Team, 2018-04-03
//
#include "hail/CommonDefs.h"
#include "hail/NativeObj.h"
#include "hail/NativeModule.h"
#include <jni.h>
#include <memory>

NAMESPACE_BEGIN(hail)

typedef long LongFuncN(...);
typedef NativeObjPtr PtrFuncN(...);

template<typename ReturnT>
class NativeFuncObj : public NativeObj {
public:
  typedef ReturnT FuncType(...);
public:
  NativeObjPtr module_; // keep-alive for the loaded module
  FuncType *func_;
  
public:
  inline NativeFuncObj(
    NativeObjPtr module,
    void* funcAddr
  ) :
    module_(module) {
    // It's awkward to turn a "void*" into a function pointer
    void** pFunc = reinterpret_cast<void**>(&func_);
    *pFunc = funcAddr;
  }
  
  virtual ~NativeFuncObj() { }

private:
  // disable copy-assign
  NativeFuncObj& operator=(const NativeFuncObj& b);
  
};

//
// Simple class to manage conversion of Java/Scala String params
//
class JString {
private:
  JNIEnv* env_;
  jstring val_;
  const char* str_;
public:
  inline JString(JNIEnv* env, jstring val) :
    env_(env),
    val_(val),
    str_(env->GetStringUTFChars(val, 0)) {
  }
  
  inline operator const char*() const {
    return(str_);
  }
  
  inline ~JString() {
    env_->ReleaseStringUTFChars(val_, str_);
  }
};

inline PtrFuncN* getPtrFuncN(long addr) {
  return reinterpret_cast< NativeFuncObj<NativeObjPtr>* >(addr)->func_;
}

inline LongFuncN* getLongFuncN(long addr) {
  return reinterpret_cast< NativeFuncObj<long>* >(addr)->func_;
}

NativeObj* getFromNativePtr(JNIEnv* env, jobject obj);

void initNativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr);

void moveToNativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr);

NAMESPACE_END(hail)

#endif
