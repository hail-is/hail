#include <jni.h>
#include <dlfcn.h>
#include <cstdio>

// Declare a function to be exported from a DLL with "C" linkage

#define NATIVEMETHOD(cppReturnType, scalaClass, scalaMethod) \
  extern "C" __attribute__((visibility("default"))) \
    cppReturnType Java_is_hail_nativecode_##scalaClass##_##scalaMethod

NATIVEMETHOD(jlong, NativeCode, dlopenGlobal)(
  JNIEnv* env,
  jobject /*thisJ*/,
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
  JNIEnv* env,
  jobject /*thisJ*/,
  jlong handle
) {
  jlong result = dlclose(reinterpret_cast<void*>(handle));
  return result;
}
