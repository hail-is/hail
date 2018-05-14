// src/main/c/NativeModule.cpp - native funcs for Scala NativeModule
#include "hail/NativeModule.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include <assert.h>
#include <jni.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <atomic>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if 0
#define D(fmt, ...) { \
  char buf[1024]; \
  sprintf(buf, fmt, ##__VA_ARGS__); \
  fprintf(stderr, "DEBUG: %s,%d: %s", __FILE__, __LINE__, buf); \
}
#else
#define D(fmt, ...) { }
#endif

namespace hail {

namespace {

// File-polling interval in usecs
const int kFilePollMicrosecs = 50000;

// A quick-and-dirty way to get a hash of two strings, take 80bits,
// and produce a 20byte string of hex digits.
std::string hash_two_strings(const std::string& a, const std::string& b) {
  auto hashA = std::hash<std::string>()(a);
  auto hashB = std::hash<std::string>()(b);
  hashA ^= (0x3ac5*hashB);
  char buf[128];
  char* out = buf;
  for (int pos = 80; (pos -= 4) >= 0;) {
    long nibble = ((pos >= 64) ? (hashB >> (pos-64)) : (hashA >> pos)) & 0xf;
    *out++ = ((nibble < 10) ? nibble+'0' : nibble-10+'a');
  }
  *out = 0;
  return std::string(buf);
}

bool file_exists_and_is_recent(const std::string& name) {
  time_t now = ::time(nullptr);
  struct stat st;
  st.st_mtime = now;
  int rc;
  do {
    errno = 0;
    rc = stat(name.c_str(), &st);
  } while ((rc < 0) && (errno == EINTR));
  return ((rc == 0) && (st.st_mtime+120 > now));
}

bool file_exists(const std::string& name) {
  int rc;
  struct stat st;
  do {
    errno = 0;
    rc = stat(name.c_str(), &st);
  } while ((rc < 0) && (errno == EINTR));
  return(rc == 0);
}

long file_size(const std::string& name) {
  int rc;
  struct stat st;
  do {
    errno = 0;
    rc = ::stat(name.c_str(), &st);
  } while ((rc < 0) && (errno = EINTR));
  return((rc < 0) ? -1 : st.st_size);
}

std::string read_file_as_string(const std::string& name) {
  FILE* f = fopen(name.c_str(), "r");
  if (!f) return std::string("");
  std::stringstream ss;
  for (;;) {
    int c = fgetc(f);
    if (c == EOF) break;
    ss << (char)c;
  }
  fclose(f);
  return ss.str();
}

std::string getenv_with_default(const char* name, const char* dval) {
  const char* s = ::getenv(name);
  return std::string(s ? s : dval);
}

std::string run_shell_get_first_line(const char* cmd) {
  FILE* f = popen(cmd, "r");
  char buf[1024];
  size_t len = 0;
  if (f) {
    for (;;) {
      int c = fgetc(f);
      if (c == EOF) break;
      if ((c != '\n') && (len+1 < sizeof(buf))) buf[len++] = c;
    }
    pclose(f);
  }
  buf[len] = 0;
  return std::string(buf);
}

std::string get_java_home() {
  auto s = run_shell_get_first_line(
    "java -XshowSettings:properties -version 2>&1 | fgrep -i java.home"
  );
  auto p = strstr(s.c_str(), "java.home = ");
  if (p) {
    return std::string(p+12);
  } else {
    return std::string(getenv_with_default("JAVA_HOME", "JAVA_HOME_undefined"));
  }
}

std::string get_cxx_name() {
  char* p = ::getenv("CXX");
  if (p) return std::string(p);
  // We prefer clang because it has faster compile
  auto s = run_shell_get_first_line("which clang");
  if (strstr(s.c_str(), "clang")) return s;
  s = run_shell_get_first_line("which g++");
  if (strstr(s.c_str(), "g++")) return s;
  // The last guess is to just say "c++"
  return std::string("c++");
}

class ModuleConfig {
public:
  bool is_darwin_;
  std::string ext_cpp_;
  std::string ext_lib_;
  std::string ext_mak_;
  std::string ext_new_;
  std::string module_dir_;
  std::string cxx_name_;
  std::string java_home_;
  std::string java_md_;

public:
  ModuleConfig() :
#if defined(__APPLE__) && defined(__MACH__)
    is_darwin_(true),
#else
    is_darwin_(false),
#endif
    ext_cpp_(".cpp"),
    ext_lib_(is_darwin_ ? ".dylib" : ".so"),
    ext_mak_(".mak"),
    ext_new_(".new"),
    module_dir_(getenv_with_default("HOME", "/tmp")+"/hail_modules"),
    cxx_name_(get_cxx_name()),
    java_home_(get_java_home()),
    java_md_(is_darwin_ ? "darwin" : "linux") {
    
  }
  
  std::string get_lib_name(const std::string& key) {
    std:: stringstream ss;
    ss << module_dir_ << "/hm_" << key  << ext_lib_;
    return ss.str();
  }
  
  std::string get_new_name(const std::string& key) {
    std:: stringstream ss;
    ss << module_dir_ << "/hm_" << key  << ext_new_;
    return ss.str();
  }
  
  void ensure_module_dir_exists() {
    int rc = ::access(module_dir_.c_str(), R_OK);
    if (rc < 0) { // create it
      rc = ::mkdir(module_dir_.c_str(), 0666);
      if (rc < 0) perror(module_dir_.c_str());
      rc = ::chmod(module_dir_.c_str(), 0755);
    }
  }
};

ModuleConfig config;

} // end anon

// ModuleBuilder deals with compiling/linking source code to a DLL,
// and providing the binary DLL as an Array[Byte] which can be broadcast
// to all workers.

class ModuleBuilder {
private:
  std::string options_;
  std::string source_;
  std::string include_;
  std::string key_;
  std::string hm_base_;
  std::string hm_mak_;
  std::string hm_cpp_;
  std::string hm_new_;
  std::string hm_lib_;
  
public:
  ModuleBuilder(
    const std::string& options,
    const std::string& source,
    const std::string& include,
    const std::string& key
  ) :
    options_(options),
    source_(source),
    include_(include),
    key_(key) {
    // To start with, put dynamic code in $HOME/hail_modules
    auto base = (config.module_dir_ + "/hm_") + key_;
    hm_base_ = base;
    hm_mak_ = (base + config.ext_mak_);
    hm_cpp_ = (base + config.ext_cpp_);
    hm_new_ = (base + config.ext_new_);
    hm_lib_ = (base + config.ext_lib_);
  }
  
  virtual ~ModuleBuilder() { }
  
private:
  void write_cpp() {
    FILE* f = fopen(hm_cpp_.c_str(), "w");
    if (!f) { perror("fopen"); return; }
    fwrite(source_.data(), 1, source_.length(), f);
    fclose(f);
  }
  
  void write_mak() {
    FILE* f = fopen(hm_mak_.c_str(), "w");
    if (!f) { perror("fopen"); return; }
    std::string javaHome = config.java_home_;
    std::string javaMD = config.java_md_;
    fprintf(f, "MODULE    := hm_%s\n", key_.c_str());
    fprintf(f, "MODULE_SO := $(MODULE)%s\n", config.ext_lib_.c_str());
    fprintf(f, "CXX       := %s\n", config.cxx_name_.c_str());
    // Downgrading from -std=c++14 to -std=c++11 for CI w/ old compilers
    const char* cxxstd = (strstr(config.cxx_name_.c_str(), "clang") ? "-std=c++17" : "-std=c++11");
    fprintf(f, "CXXFLAGS  := \\\n");
    fprintf(f, "  %s -fPIC -march=native -fno-strict-aliasing -Wall -Werror \\\n", cxxstd);
    fprintf(f, "  -I%s/include \\\n", config.java_home_.c_str());
    fprintf(f, "  -I%s/include/%s \\\n", config.java_home_.c_str(), config.java_md_.c_str());
    fprintf(f, "  -I%s \\\n", include_.c_str());
    bool have_oflag = (strstr(options_.c_str(), "-O") != nullptr);
    fprintf(f, "  %s%s \\\n", have_oflag ? "" : "-O3", options_.c_str());
    fprintf(f, "  -DHAIL_MODULE=$(MODULE)\n");
    fprintf(f, "LIBFLAGS := -fvisibility=default %s\n", 
      config.is_darwin_ ? "-dynamiclib -Wl,-undefined,dynamic_lookup"
                         : "-rdynamic -shared");
    fprintf(f, "\n");
    // top target is the .so
    fprintf(f, "$(MODULE_SO): $(MODULE).o\n");
    fprintf(f, "\t/bin/mv -f $(MODULE).new $@\n\n");
    // build .o from .cpp
    fprintf(f, "$(MODULE).o: $(MODULE).cpp\n");
    fprintf(f, "\t$(CXX) $(CXXFLAGS) -o $@ -c $< 2> $(MODULE).err \\\n");
    fprintf(f, "\t  || ( /bin/rm -f $(MODULE).new ; exit 1 )\n");
    fprintf(f, "\t$(CXX) $(CXXFLAGS) $(LIBFLAGS) -o $(MODULE).new $(MODULE).o\n");
    fprintf(f, "\t/bin/chmod a+rx $(MODULE).new\n");
    fprintf(f, "\t/bin/rm -f $(MODULE).err\n\n");
    fclose(f);
  }

public:
  bool try_to_start_build() {
    // Try to create the .new file
    FILE* f = fopen(hm_new_.c_str(), "w+");
    if (!f) {
      // We lost the race to start the build
      return false;
    }
    fclose(f);
    // The .new file may look the same age as the .cpp file, but
    // the makefile is written to ignore the .new timestamp
    write_mak();
    write_cpp();
    std::stringstream ss;
    // ss << "/usr/bin/nohup ";
    ss << "/usr/bin/make -C " << config.module_dir_ << " -f " << hm_mak_;
    ss << " >/dev/null &";
    int rc = system(ss.str().c_str());
    if (rc < 0) perror("system");
    return true;
  }
};

NativeModule::NativeModule(
  const char* options,
  const char* source,
  const char* include,
  bool force_build
) :
  build_state_(kInit),
  load_state_(kInit),
  key_(hash_two_strings(options, source)),
  is_global_(false),
  dlopen_handle_(nullptr),
  lib_name_(config.get_lib_name(key_)),
  new_name_(config.get_new_name(key_)) {
  // Master constructor - try to get module built in local file
  config.ensure_module_dir_exists();
  if (!force_build && file_exists(lib_name_)) {
    build_state_ = kPass;
  } else {
    // The file doesn't exist, let's start building it
    ModuleBuilder builder(options, source, include, key_);
    builder.try_to_start_build();
  }
}

NativeModule::NativeModule(
  bool is_global,
  const char* key,
  long binary_size,
  const void* binary
) :
  build_state_(is_global ? kPass : kInit),
  load_state_(is_global ? kPass : kInit),
  key_(key),
  is_global_(is_global),
  dlopen_handle_(nullptr),
  lib_name_(config.get_lib_name(key_)),
  new_name_(config.get_new_name(key_)) {
  // Worker constructor - try to get the binary written to local file
  if (is_global_) return;
  int rc = 0;
  config.ensure_module_dir_exists();
  for (;;) {
    if (file_exists(lib_name_) && (file_size(lib_name_) == binary_size)) {
      build_state_ = kPass;
      break;
    }
    // Race to write the new file
    int fd = open(new_name_.c_str(), O_WRONLY|O_CREAT|O_EXCL|O_TRUNC, 0666);
    if (fd >= 0) {
      // Now we're about to write the new file
      rc = write(fd, binary, binary_size);
      assert(rc == binary_size);
      close(fd);
      ::chmod(new_name_.c_str(), 0644);
      if (!file_exists(lib_name_)) {
        // Don't let anyone see the file until it is completely written
        rc = ::rename(new_name_.c_str(), lib_name_.c_str());
        build_state_ = ((rc == 0) ? kPass : kFail);
        break;
      }
    } else {
      // Someone else is writing to new
      while (file_exists_and_is_recent(new_name_) && !file_exists(lib_name_)) {
        usleep(kFilePollMicrosecs);
      }
    }
  }
  if (build_state_ == kPass) try_load();
}

NativeModule::~NativeModule() {
  if (!is_global_ && dlopen_handle_) {
    dlclose(dlopen_handle_);
  }
}

bool NativeModule::try_wait_for_build() {
  if (build_state_ == kInit) {
    // The writer will rename new to lib.  If we tested exists(lib)
    // followed by exists(new) then the rename could occur between
    // the two tests. This way is safe provided that either rename is atomic,
    // or rename creates the new name before destroying the old name.
    while (file_exists_and_is_recent(new_name_)) {
      usleep(kFilePollMicrosecs);
    }
    build_state_ = (file_exists(lib_name_) ? kPass : kFail);
    if (build_state_ == kFail) {
      std::string base(config.module_dir_ + "/hm_" + key_);
      fprintf(stderr, "makefile:\n%s", read_file_as_string(base+".mak").c_str());
      fprintf(stderr, "errors:\n%s",   read_file_as_string(base+".err").c_str());
    }
  }
  return(build_state_ == kPass);
}

bool NativeModule::try_load() {
  if (load_state_ == kInit) {
    if (is_global_) {
      load_state_ = kPass;
    } else if (!try_wait_for_build()) {
      fprintf(stderr, "libName %s try_wait_for_build fail\n", lib_name_.c_str());
      load_state_ = kFail;
    } else {
      auto handle = dlopen(lib_name_.c_str(), RTLD_GLOBAL|RTLD_LAZY);
      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
      }
      load_state_ = (handle ? kPass : kFail);
      if (handle) dlopen_handle_ = handle;
    }
  }
  return(load_state_ == kPass);
}

static std::string to_qualified_name(
  JNIEnv* env,
  const std::string& key,
  jstring nameJ,
  int numArgs,
  bool is_global
) {
  JString name(env, nameJ);
  char argTypeCodes[32];
  for (int j = 0; j < numArgs; ++j) argTypeCodes[j] = 'l';
  argTypeCodes[numArgs] = 0;  
  char buf[512];
  if (is_global) {
    // No name-mangling for global func names
    strcpy(buf, name);
  } else {
    auto moduleName = std::string("hm_") + key;
    sprintf(buf, "_ZN4hail%lu%s%lu%sE%s",
      moduleName.length(), moduleName.c_str(), strlen(name), (const char*)name, argTypeCodes);
  }
  return std::string(buf);
}

void NativeModule::find_LongFuncL(
  JNIEnv* env,
  NativeStatus* st,
  jobject funcObj,
  jstring nameJ,
  int numArgs
) {
  void* funcAddr = nullptr;
  if (!try_load()) {
    NATIVE_ERROR(st, 1001, "ErrModuleNotFound");
  } else {
    auto qualName = to_qualified_name(env, key_, nameJ, numArgs, is_global_);
    D("is_global %s qualName \"%s\"\n", is_global_ ? "true" : "false", qualName.c_str());    
    funcAddr = ::dlsym(is_global_ ? RTLD_DEFAULT : dlopen_handle_, qualName.c_str());
    D("dlsym -> funcAddr %p\n", funcAddr);
    if (!funcAddr) {
      NATIVE_ERROR(st, 1003, "ErrLongFuncNotFound dlsym(\"%s\")", qualName.c_str());
    }
  }
  auto ptr = MAKE_NATIVE(NativeFuncObj<long>, shared_from_this(), funcAddr);
  init_NativePtr(env, funcObj, &ptr);
}

void NativeModule::find_PtrFuncL(
  JNIEnv* env,
  NativeStatus* st,
  jobject funcObj,
  jstring nameJ,
  int numArgs
) {
  void* funcAddr = nullptr;
  if (!try_load()) {
    NATIVE_ERROR(st, 1001, "ErrModuleNotFound");
  } else {
    auto qualName = to_qualified_name(env, key_, nameJ, numArgs, is_global_);
    funcAddr = ::dlsym(is_global_ ? RTLD_DEFAULT : dlopen_handle_, qualName.c_str());
    if (!funcAddr) {
      NATIVE_ERROR(st, 1003, "ErrPtrFuncNotFound dlsym(\"%s\")", qualName.c_str());
    }
  }
  auto ptr = MAKE_NATIVE(NativeFuncObj<NativeObjPtr>, shared_from_this(), funcAddr);
  init_NativePtr(env, funcObj, &ptr);
}

// Functions implementing NativeModule native methods

static NativeModule* to_NativeModule(JNIEnv* env, jobject obj) {
  // It should be a dynamic_cast, but I'm trying to eliminate
  // the use of RTTI which is problematic in dynamic libraries
  return reinterpret_cast<NativeModule*>(get_from_NativePtr(env, obj));
}

NATIVEMETHOD(void, NativeModule, nativeCtorMaster)(
  JNIEnv* env,
  jobject thisJ,
  jstring optionsJ,
  jstring sourceJ,
  jstring includeJ,
  jboolean force_buildJ
) {
  JString options(env, optionsJ);
  JString source(env, sourceJ);
  JString include(env, includeJ);
  bool force_build = (force_buildJ != JNI_FALSE);
  auto ptr = MAKE_NATIVE(NativeModule, options, source, include, force_build);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativeModule, nativeCtorWorker)(
  JNIEnv* env,
  jobject thisJ,
  jboolean is_globalJ,
  jstring keyJ,
  jbyteArray binaryJ
) {
  bool is_global = (is_globalJ != JNI_FALSE);
  JString key(env, keyJ);
  long binary_size = env->GetArrayLength(binaryJ);
  auto binary = env->GetByteArrayElements(binaryJ, 0);
  auto ptr = MAKE_NATIVE(NativeModule, is_global, key, binary_size, binary);
  env->ReleaseByteArrayElements(binaryJ, binary, JNI_ABORT);
  init_NativePtr(env, thisJ, &ptr);
}

NATIVEMETHOD(void, NativeModule, nativeFindOrBuild)(
  JNIEnv* env,
  jobject thisJ,
  long stAddr
) {
  auto mod = to_NativeModule(env, thisJ);
  auto st = reinterpret_cast<NativeStatus*>(stAddr);
  st->clear();
  if (!mod->try_wait_for_build()) {
    NATIVE_ERROR(st, 1004, "ErrModuleBuildFailed");
  }
}

NATIVEMETHOD(jstring, NativeModule, getKey)(
  JNIEnv* env,
  jobject thisJ
) {
  auto mod = to_NativeModule(env, thisJ);
  return env->NewStringUTF(mod->key_.c_str());
}

NATIVEMETHOD(jbyteArray, NativeModule, getBinary)(
  JNIEnv* env,
  jobject thisJ
) {
  auto mod = to_NativeModule(env, thisJ);
  int fd = open(config.get_lib_name(mod->key_).c_str(), O_RDONLY, 0666);
  if (fd < 0) {
    perror("open");
    return env->NewByteArray(0);
  }
  struct stat st;
  int rc = fstat(fd, &st);
  assert(rc == 0);
  size_t file_size = st.st_size;
  jbyteArray result = env->NewByteArray(file_size);
  jbyte* rbuf = env->GetByteArrayElements(result, 0);
  rc = read(fd, rbuf, file_size);
  assert(rc == (int)file_size);
  close(fd);
  env->ReleaseByteArrayElements(result, rbuf, 0);
  return result;
}

#define DECLARE_FIND(LongOrPtr, num_args) \
NATIVEMETHOD(void, NativeModule, nativeFind##LongOrPtr##FuncL##num_args)( \
  JNIEnv* env, \
  jobject thisJ, \
  long stAddr, \
  jobject funcJ, \
  jstring nameJ \
) { \
  auto mod = to_NativeModule(env, thisJ); \
  auto st = reinterpret_cast<NativeStatus*>(stAddr); \
  st->clear(); \
  mod->find_##LongOrPtr##FuncL(env, st, funcJ, nameJ, num_args); \
}

DECLARE_FIND(Long, 0)
DECLARE_FIND(Long, 1)
DECLARE_FIND(Long, 2)
DECLARE_FIND(Long, 3)
DECLARE_FIND(Long, 4)
DECLARE_FIND(Long, 5)
DECLARE_FIND(Long, 6)
DECLARE_FIND(Long, 7)
DECLARE_FIND(Long, 8)

DECLARE_FIND(Ptr, 0)
DECLARE_FIND(Ptr, 1)
DECLARE_FIND(Ptr, 2)
DECLARE_FIND(Ptr, 3)
DECLARE_FIND(Ptr, 4)
DECLARE_FIND(Ptr, 5)
DECLARE_FIND(Ptr, 6)
DECLARE_FIND(Ptr, 7)
DECLARE_FIND(Ptr, 8)

} // end hail
