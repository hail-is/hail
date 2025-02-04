#include <jni.h>
#include <dlfcn.h>
#include <cstdio>
#include "hail/NativeMethod.h"

NATIVEMETHOD(jlong, NativeCode, dlopenGlobal)(
  JNIEnv* env,
  jobject,
  jstring dllPathJ
) {
  const char* dll_path = env->GetStringUTFChars(dllPathJ, 0);
  void* handle = dlopen(dll_path, RTLD_GLOBAL|RTLD_LAZY);
  if (!handle) {
    char* msg = dlerror();
    fprintf(stderr, "ERROR: dlopen(\"%s\"): %s\n", dll_path, msg ? msg : "NoError");
  }
  env->ReleaseStringUTFChars(dllPathJ, dll_path);
  return reinterpret_cast<jlong>(handle);
}

NATIVEMETHOD(jlong, NativeCode, dlclose)(
  JNIEnv*,
  jobject,
  jlong handle
) {
  jlong result = dlclose(reinterpret_cast<void*>(handle));
  return result;
}
