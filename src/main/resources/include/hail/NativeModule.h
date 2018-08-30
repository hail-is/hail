#ifndef HAIL_NATIVEMODULE_H
#define HAIL_NATIVEMODULE_H 1

#include "hail/NativeObj.h"
#include "hail/NativeStatus.h"
#include <jni.h>
#include <cstdint>
#include <string>
#include <vector>

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
  NativeModule(const char* options, const char* source, const char* include);
  
  NativeModule(bool isGlobal, const char* key, ssize_t binarySize, const void* binary);

  virtual ~NativeModule();
  
  virtual const char* get_class_name() {
    return "NativeModule";
  }
  
  std::vector<char> get_binary();
  
  void find_LongFuncL(JNIEnv* env, NativeStatus* st, jobject funcObj, jstring nameJ, int numArgs);

  void find_PtrFuncL(JNIEnv* env, NativeStatus* st, jobject funcObj, jstring nameJ, int numArgs);

  NativeModule(const NativeModule& b) = delete;

  NativeModule& operator=(const NativeModule& b) = delete;

  // Methods with names ending "_locked" must be called already holding the big_mutex

  bool try_load_locked();  
};

// Each NativeFunc or NativeMaker holds a NativeModulePtr
// to ensure that the module can't be dlclose'd while
// its func's might still be called.
using NativeModulePtr = std::shared_ptr<NativeModule>;

} // end hail

#endif
