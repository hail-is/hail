#ifndef HAIL_NATIVEMODULE_H
#define HAIL_NATIVEMODULE_H 1
//
// Declarations corresponding to NativeModule.scala
//
// Richard Cownie, Hail Team, 2018-04-12
//
#include "hail/CommonDefs.h"
#include "hail/NativeObj.h"
#include "hail/NativeStatus.h"
#include <jni.h>
#include <string>

NAMESPACE_BEGIN(hail)

//
// Off-heap object referenced by Scala NativeModule
//
class NativeModule : public NativeObj {
public:
  enum State { kInit, kPass, kFail };
public:
  State buildState_;
  State loadState_;
  std::string key_;
  bool isGlobal_;
  void* dlopenHandle_;
  std::string libName_;
  std::string newName_;
  
public:
  NativeModule(const char* options, const char* source, bool forceBuild);
  
  NativeModule(bool isGlobal, const char* key, long binarySize, const void* binary);

  virtual ~NativeModule();
  
  virtual const char* getClassName() {
    return "NativeModule";
  }
  
  bool tryWaitForBuild();
  
  bool tryLoad();
  
  void findLongFuncL(JNIEnv* env, NativeStatus* st, jobject funcObj, jstring nameJ, int numArgs);

  void findPtrFuncL(JNIEnv* env, NativeStatus* st, jobject funcObj, jstring nameJ, int numArgs);

private:
  // disallow copy-construct
  NativeModule(const NativeModule& b);
  // disallow copy-assign
  NativeModule& operator=(const NativeModule& b);
};

//
// Each NativeFunc or NativeMaker holds a NativeModulePtr
// to ensure that the module can't be dlclose'd while
// its func's might still be called.
//
typedef std::shared_ptr<NativeModule> NativeModulePtr;

NAMESPACE_END(hail)

#endif
