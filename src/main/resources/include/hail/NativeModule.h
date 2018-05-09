#ifndef HAIL_NATIVEMODULE_H
#define HAIL_NATIVEMODULE_H 1

#include "hail/NativeObj.h"
#include "hail/NativeStatus.h"
#include <jni.h>
#include <string>

namespace hail {

// Off-heap object referenced by Scala NativeModule

class NativeModule : public NativeObj {
public:
  enum State { kInit, kPass, kFail };
public:
  State build_state_;
  State load_state_;
  std::string key_;
  bool is_global_;
  void* dlopen_handle_;
  std::string lib_name_;
  std::string new_name_;
  
public:
  NativeModule(const char* options, const char* source, const char* include, bool forceBuild);
  
  NativeModule(bool isGlobal, const char* key, long binarySize, const void* binary);

  virtual ~NativeModule();
  
  virtual const char* get_class_name() {
    return "NativeModule";
  }
  
  bool try_wait_for_build();
  
  bool try_load();
  
  void find_LongFuncL(JNIEnv* env, NativeStatus* st, jobject funcObj, jstring nameJ, int numArgs);

  void find_PtrFuncL(JNIEnv* env, NativeStatus* st, jobject funcObj, jstring nameJ, int numArgs);

private:
  // disallow copy-construct
  NativeModule(const NativeModule& b);
  // disallow copy-assign
  NativeModule& operator=(const NativeModule& b);
};

// Each NativeFunc or NativeMaker holds a NativeModulePtr
// to ensure that the module can't be dlclose'd while
// its func's might still be called.
using NativeModulePtr = std::shared_ptr<NativeModule>;

} // end hail

#endif
