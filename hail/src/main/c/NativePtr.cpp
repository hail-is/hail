#include "hail/NativeMethod.h"
#include "hail/NativePtr.h"
#include "hail/NativeObj.h"
#include "hail/hail.h"
#include <cassert>
#include <csignal>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <jni.h>
#include <execinfo.h>
#include <unistd.h>

extern "C" int hail_gdb_breakpoint() {
  // GdbConfig sets a breakpoint here
  static int count = 0;
  return ++count;;
}

#ifdef HAIL_ENABLE_DEBUG
extern "C" void hail_pause_for_gdb(const char* file, int line, const char* why) {
  fprintf(stderr, "DEBUG: %s,%d: HAIL_PAUSE %s ...\n", file, line, why);
  int n = hail_gdb_breakpoint();
  fprintf(stderr, "DEBUG: %s,%d: HAIL_PAUSE %s done (count %d)\n", file, line, why, n);
}
#else
extern "C" void hail_pause_for_gdb(const char*, int, const char*) { }
#endif

namespace hail {

// JNI interface functions corresponding to Scala NativeBase and NativePtr

#define NATIVEFUNC(typ) \
  extern "C" JNIEXPORT typ JNICALL

namespace {

class GdbConfig {
 public:
  GdbConfig() {
#ifdef HAIL_ENABLE_DEBUG
    fprintf(stderr, "DEBUG: trying to run gdb under xterm ...\n");
    FILE* f = fopen("a.gdb", "w");
    fprintf(f, "handle SIGSEGV noprint\n");
    fprintf(f, "handle SIG62 noprint\n");
    fprintf(f, "break hail_gdb_breakpoint\n");
    fclose(f);
    int pid = getpid();
    char cmd[512];
    sprintf(
      cmd,
      "xterm -rightbar -e \"sudo gdb -p %d -x a.gdb\" &\n",
      pid
    );
    int rc = system(cmd);
    fprintf(stderr, "DEBUG: system(\"%s\") -> %d\n", cmd, rc);
    for (int j = 10; j >= 0; --j) {
      fprintf(stderr, "%d", j);
      if (j > 0) {
        fprintf(stderr, "..");
        sleep(1);
      }
    }
    fprintf(stderr, " continue\n");
#endif
  }
};

GdbConfig gdb_config;

void check_assumptions();

void on_signal(int sig) {
  const char* signame = "UNKNOWN";
  if      (sig == SIGSEGV) signame = "SIGSEGV";
  else if (sig == SIGFPE)  signame = "SIGFPE";
  else if (sig == SIGBUS)  signame = "SIGBUS";
  else if (sig == SIGILL)  signame = "SIGILL";
  else if (sig == SIGABRT) signame = "SIGABRT";
  fprintf(stderr, "FATAL: caught signal %d %s\n", sig, signame);
  void* callstack[20];
  int depth = backtrace(callstack, 20);
  backtrace_symbols_fd(callstack, depth, 2);
}

void catch_signals() {
  signal(SIGABRT, on_signal);
  //signal(SIGBUS,  on_signal);
  signal(SIGFPE,  on_signal);
  signal(SIGILL,  on_signal);
  //signal(SIGSEGV, on_signal);
}

class NativePtrInfo {
public:
  JavaVM* java_vm_;
  jclass class_ref_;
  jfieldID addrA_id_;
  jfieldID addrB_id_;

public:
  NativePtrInfo(JNIEnv* env, int) {
    auto rc = env->GetJavaVM(&java_vm_); // needed for making C++-to-JVM calls
    assert(rc == JNI_OK);
    auto cl = env->FindClass("is/hail/nativecode/NativeBase");
    class_ref_ = (jclass)env->NewGlobalRef(cl);
    env->DeleteLocalRef(cl);
    addrA_id_ = env->GetFieldID(class_ref_, "addrA", "J");
    addrB_id_ = env->GetFieldID(class_ref_, "addrB", "J");
    check_assumptions();
    catch_signals();
  }
};

// WARNING: I observe that when this gets loaded as a shared library on Linux,
// we see two distinct NativePtrInfo objects at different addresses.  That is
// extremely weird, but by putting all the initialization into the constructor
// we make sure that both get correctly initialized.  But that behavior might
// cause trouble in other code, so we need to watch out for it.
//
// I suspect this may be related to gcc's use of STB_GNU_UNIQUE for static
// data, though from this description it shouldn't be used for data in a
// non-inline function:
//
// "On systems with recent GNU assembler and C library, the C++ compiler uses
// the STB_GNU_UNIQUE binding to make sure that definitions of template static
// data members and static local variables in inline functions are unique even
// in the presence of RTLD_LOCAL"


// We could in theory do the initialization from a JNI_OnLoad() function, but
// don't want to take time experimenting with that now.

static NativePtrInfo* get_info(JNIEnv* env, int line) {
  static NativePtrInfo the_info(env, line);
  return &the_info;
}

// We use this class for moving between a genuine std::shared_ptr<T>,
// and the Scala NativeBase addrA, addrB.  Sometimes we have a temporary
// TwoAddrs which we temporarily view as NativeObjPtr; and sometimes we
// have a genuine NativeObjPtr which we temporarily view as TwoAddrs.

class TwoAddrs {
public:
  int64_t addrs_[2];

public:
  TwoAddrs(JNIEnv* env, jobject obj, NativePtrInfo* info) {
    addrs_[0] = env->GetLongField(obj, info->addrA_id_);
    addrs_[1] = env->GetLongField(obj, info->addrB_id_);
  }

  TwoAddrs(int64_t addrA, int64_t addrB) {
    addrs_[0] = addrA;
    addrs_[1] = addrB;
  }

  TwoAddrs(NativeObjPtr&& b) {
    addrs_[0] = 0;
    addrs_[1] = 0;
    this->as_NativeObjPtr() = std::move(b);
  }

  TwoAddrs(const NativeObjPtr& b) {
    addrs_[0] = 0;
    addrs_[1] = 0;
    this->as_NativeObjPtr() = b;
  }

  NativeObjPtr& as_NativeObjPtr() {
    return *reinterpret_cast<NativeObjPtr*>(this);
  }

  void copy_to_scala(JNIEnv* env, jobject obj, NativePtrInfo* info) {
    env->SetLongField(obj, info->addrA_id_, addrs_[0]);
    env->SetLongField(obj, info->addrB_id_, addrs_[1]);
  }

  int64_t get_addrA() const { return addrs_[0]; }
  int64_t get_addrB() const { return addrs_[1]; }
  void set_addrA(int64_t v) { addrs_[0] = v; }
  void set_addrB(int64_t v) { addrs_[1] = v; }
};

void check_assumptions() {
  // Check that std::shared_ptr matches our assumptions
  auto ptr = std::make_shared<NativeObj>();
  TwoAddrs* buf = reinterpret_cast<TwoAddrs*>(&ptr);
  assert(sizeof(ptr) == 2*sizeof(int64_t));
  assert(buf->get_addrA() == reinterpret_cast<int64_t>(ptr.get()));
}

} // end anon

JavaVM* get_saved_java_vm() {
  // This will *not* be the first call to get_info(), so we don't need
  // a valid JNIEnv*
  return get_info(nullptr, 0)->java_vm_;
}

NativeObj* get_from_NativePtr(JNIEnv* env, jobject obj) {
  auto info = get_info(env, __LINE__);
  int64_t addrA = env->GetLongField(obj, info->addrA_id_);
  return reinterpret_cast<NativeObj*>(addrA);
}

void init_NativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  auto info = get_info(env, __LINE__);
  // Ignore previous values in NativePtr
  TwoAddrs buf(std::move(*ptr));
  buf.copy_to_scala(env, obj, info);
}

void move_to_NativePtr(JNIEnv* env, jobject obj, NativeObjPtr* ptr) {
  auto info = get_info(env, __LINE__);
  TwoAddrs buf(env, obj, info);
  buf.as_NativeObjPtr() = std::move(*ptr);
  buf.copy_to_scala(env, obj, info);
}

NATIVEMETHOD(void, NativeBase, nativeCopyCtor)(
  JNIEnv* env,
  jobject thisJ,
  jlong b_addrA,
  jlong
) {
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
  auto info = get_info(env, __LINE__);
  TwoAddrs bufA(env, thisJ, info);
  TwoAddrs bufB(env, srcJ, info);
  bufA.as_NativeObjPtr() = bufB.as_NativeObjPtr();
  bufA.copy_to_scala(env, thisJ, info);
}

NATIVEMETHOD(void, NativeBase, moveAssign)(
  JNIEnv* env,
  jobject thisJ,
  jobject srcJ
) {
  auto info = get_info(env, __LINE__);
  if (thisJ == srcJ) return;
  TwoAddrs bufA(env, thisJ, info);
  TwoAddrs bufB(env, srcJ, info);
  bufA.as_NativeObjPtr() = std::move(bufB.as_NativeObjPtr());
  bufA.copy_to_scala(env, thisJ, info);
  bufB.copy_to_scala(env, srcJ, info);
}

NATIVEMETHOD(void, NativeBase, nativeReset)(
  JNIEnv*,
  jobject,
  jlong addrA,
  jlong addrB
) {
  TwoAddrs bufA(addrA, addrB);
  bufA.as_NativeObjPtr().reset();
  // The Scala object fields are cleared in the wrapper
}

NATIVEMETHOD(jlong, NativeBase, nativeUseCount)(
  JNIEnv*,
  jobject,
  jlong addrA,
  jlong addrB
) {
  TwoAddrs bufA(addrA, addrB);
  return bufA.as_NativeObjPtr().use_count();
}

// We have constructors corresponding to std::make_shared<T>(...)
// with various numbers of int64_t arguments.

inline NativeStatus* to_status(jlong st) {
  return reinterpret_cast<NativeStatus*>(st);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL0)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st));
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL1)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL2)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL3)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1, a2);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL4)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1, a2, a3);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL5)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1, a2, a3, a4);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL6)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1, a2, a3, a4, a5);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL7)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1, a2, a3, a4, a5, a6);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativePtr, nativePtrFuncL8)(
  JNIEnv* env,
  jobject thisJ,
  jlong funcObjAddr,
  jlong st,
  jlong a0,
  jlong a1,
  jlong a2,
  jlong a3,
  jlong a4,
  jlong a5,
  jlong a6,
  jlong a7
) {
  NativeObjPtr ptr = get_PtrFuncN(funcObjAddr)(to_status(st), a0, a1, a2, a3, a4, a5, a6, a7);
  init_NativePtr(env, thisJ, &ptr);
}

// Test code for nativePtrFunc

class TestObjA : public hail::NativeObj {
private:
  std::vector<int64_t> vec_;

public:
  TestObjA() : vec_() {
    fprintf(stderr, "DEBUG: TestObjA::ctor()\n");
  }

  TestObjA(int64_t a0) : vec_({a0}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx)\n", (long)a0);
  }

  TestObjA(int64_t a0, int64_t a1) : vec_({a0, a1}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx, %08lx)\n", (long)a0, (long)a1);
  }

  TestObjA(int64_t a0, int64_t a1, int64_t a2) : vec_({a0,a1,a2}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx, %08lx, %08lx)\n", (long)a0, (long)a1, (long)a2);
  }

  TestObjA(int64_t a0, int64_t a1, int64_t a2, int64_t a3) : vec_({a0,a1,a2,a3}) {
    fprintf(stderr, "DEBUG: TestObjA::ctor(%08lx, %08lx, %08lx, %08lx)\n",
      (long)a0, (long)a1, (long)a2, (long)a3);
  }

  virtual ~TestObjA() {
    fprintf(stderr, "DEBUG: TestObjA::dtor() size %lu\n", vec_.size());
  }
};

NativeObjPtr nativePtrFuncTestObjA0(NativeStatus*) {
  return std::make_shared<TestObjA>();
}

NativeObjPtr nativePtrFuncTestObjA1(NativeStatus*, int64_t a0) {
  return std::make_shared<TestObjA>(a0);
}

NativeObjPtr nativePtrFuncTestObjA2(NativeStatus*, int64_t a0, int64_t a1) {
  return std::make_shared<TestObjA>(a0, a1);
}

NativeObjPtr nativePtrFuncTestObjA3(NativeStatus*, int64_t a0, int64_t a1, int64_t a2) {
  return std::make_shared<TestObjA>(a0, a1, a2);
}

NativeObjPtr nativePtrFuncTestObjA4(NativeStatus*, int64_t a0, int64_t a1, int64_t a2, int64_t a3) {
  return std::make_shared<TestObjA>(a0, a1, a2, a3);
}

} // end hail

