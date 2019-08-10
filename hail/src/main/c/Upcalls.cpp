#include "hail/Upcalls.h"
#include "hail/NativePtr.h"
#include <jni.h>
#include <cassert>
#include <cstdio>
#include <mutex>
#include <string>

namespace hail {

UpcallConfig::UpcallConfig() {    
  java_vm_ = get_saved_java_vm();
  JNIEnv* env = nullptr;
  auto rc = java_vm_->GetEnv((void**)&env, JNI_VERSION_1_8);
  assert(rc == JNI_OK);
  auto cl0 = env->FindClass("is/hail/nativecode/Upcalls");
  auto init_method = env->GetMethodID(cl0, "<init>", "()V");
  // NewObject gives a local ref only valid during this downcall
  auto local_upcalls = env->NewObject(cl0, init_method);
  // Get a global ref to the new object
  upcalls_ = env->NewGlobalRef(local_upcalls);
  // Java method signatures are described here:
  // http://journals.ecs.soton.ac.uk/java/tutorial/native1.1/implementing/method.html
  // "javap -v ClassName" can be used to show method signatures
  Upcalls_setTestMsg_ = env->GetMethodID(cl0, "setTestMsg", "(Ljava/lang/String;)V");
  Upcalls_info_  = env->GetMethodID(cl0, "info", "(Ljava/lang/String;)V");
  Upcalls_warn_  = env->GetMethodID(cl0, "warn", "(Ljava/lang/String;)V");
  Upcalls_error_ = env->GetMethodID(cl0, "error", "(Ljava/lang/String;)V");
  // InputStream methods
  auto cl1 = env->FindClass("java/io/InputStream");
  InputStream_close_ = env->GetMethodID(cl1, "close", "()V");
  InputStream_read_  = env->GetMethodID(cl1, "read", "([BII)I");
  InputStream_skip_  = env->GetMethodID(cl1, "skip", "(J)J");
  // InputStream methods
  auto cl3 = env->FindClass("java/io/OutputStream");
  OutputStream_close_ = env->GetMethodID(cl3, "close", "()V");
  OutputStream_flush_ = env->GetMethodID(cl3, "flush", "()V");
  OutputStream_write_ = env->GetMethodID(cl3, "write", "([BII)V");
  // RegionValueIterator methods
  auto cl4 = env->FindClass("is/hail/cxx/RegionValueIterator");
  RVIterator_hasNext_ = env->GetMethodID(cl4, "hasNext", "()Z");
  RVIterator_next_ = env->GetMethodID(cl4, "next", "()J");
}

namespace {

UpcallConfig* get_config() {
  static UpcallConfig config;
  return &config;
}

} // end anon

// Code for JNI interaction is adapted from here:
// https://stackoverflow.com/questions/30026030/what-is-the-best-way-to-save-jnienv/30026231#30026231

// At the moment the C++ does not create any threads, so all threads *should*
// be attached to the vm.  But in case that changes in future, we
// have code to handle the general case.

UpcallEnv::UpcallEnv() :
  config_(get_config()),
  env_(nullptr),
  did_attach_(false) {
  // Is this thread already attached to vm ?
  // The version checks that the running JVM version is compatible
  // with the features used in this code.
  auto vm = config_->java_vm_;
  auto rc = vm->GetEnv((void**)&env_, JNI_VERSION_1_8);
  if (rc == JNI_EDETACHED) {
    if (vm->AttachCurrentThread((void**)&env_, nullptr) != JNI_OK) {
      fprintf(stderr, "FATAL: vm->AttachCurrentThread() failed\n");
      assert(0);
    }
    did_attach_ = true;
  } else if (rc == JNI_EVERSION) {
    fprintf(stderr, "FATAL: vm->GetEnv() JNI_VERSION_1_8 not supported\n");
    assert(0);
  }
}

UpcallEnv::~UpcallEnv() {
  if (did_attach_) config_->java_vm_->DetachCurrentThread();
}

// set_test_msg is plumbed through to Scala in exactly the same way as the
// info/warn/error methods, but it saves the msg in Upcalls.testMsg
// where we can verify that the upcall delivered it correctly.

void UpcallEnv::set_test_msg(const std::string& msg) {
  jstring msgJ = env_->NewStringUTF(msg.c_str());
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_setTestMsg_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

// Logging

void UpcallEnv::info(const std::string& msg) {
  jstring msgJ = env_->NewStringUTF(msg.c_str());
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_info_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

void UpcallEnv::warn(const std::string& msg) {
  jstring msgJ = env_->NewStringUTF(msg.c_str());
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_warn_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

void UpcallEnv::error(const std::string& msg) {
  jstring msgJ = env_->NewStringUTF(msg.c_str());
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_error_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

} // namespace hail
