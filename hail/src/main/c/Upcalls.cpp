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
  // InputBuffer method[s (minimal set)
  auto cl2 = env->FindClass("is/hail/io/InputBuffer");
  InputBuffer_close_ = env->GetMethodID(cl2, "close", "()V");
  InputBuffer_readToEndOfBlock_ = env->GetMethodID(cl2, "readToEndOfBlock", "(J[BII)I");
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

void UpcallEnv::set_test_msg(const char* msg) {
  jstring msgJ = env_->NewStringUTF(msg);
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_setTestMsg_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

// Logging

void UpcallEnv::info(const char* msg) {
  jstring msgJ = env_->NewStringUTF(msg);
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_info_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

void UpcallEnv::warn(const char* msg) {
  jstring msgJ = env_->NewStringUTF(msg);
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_warn_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

void UpcallEnv::error(const char* msg) {
  jstring msgJ = env_->NewStringUTF(msg);
  env_->CallVoidMethod(config_->upcalls_, config_->Upcalls_error_, msgJ);
  env_->DeleteLocalRef(msgJ);
}

// InputBuffer

void UpcallEnv::InputBuffer_close(jobject obj) {
  env_->CallVoidMethod(obj, config_->InputBuffer_close_);
}

int32_t UpcallEnv::InputBuffer_readToEndOfBlock(
  jobject obj,
  void* toAddr,
  jbyteArray buf,
  int32_t off,
  int32_t n
) {
  return env_->CallIntMethod(obj, config_->InputBuffer_readToEndOfBlock_,
                             (jlong)toAddr, buf, (jint)off, (jint)n);
}

} // namespace hail
