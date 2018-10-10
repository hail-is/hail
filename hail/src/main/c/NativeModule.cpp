#include "hail/NativeModule.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include "MurmurHash3.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace hail {

namespace {

// Top-level NativeModule methods lock this mutex.  Constructors, and helper methods
// with names ending in "_locked", must be called only while holding the mutex.
//
// That makes everything single-threaded.

std::mutex big_mutex;

// We have a short string "options" and a long string "source"
// Produce an 80bit hash from the pair of string,
// together with a checksum of all header files tar'ed together, so that
// any change to header files forces recompilation.
//
// The shorter string (corresponding to options), may have only a
// few distinct values, so we need to mix it up with the longer
// string in various ways.
//
// We use the open-source 128-bit MurmurHash3 as a robust and fast
// hash function with no copyright restrictions.

hail::hstring hash_two_strings(const hail::hstring& a, const hail::hstring& b) {
  uint32_t seed = ALL_HEADER_CKSUM; // checksum from all header files
  uint64_t hashA[2];
  MurmurHash3_x64_128(a.c_str(), a.length(), seed, hashA);
  uint64_t hashB[2];
  MurmurHash3_x64_128(b.c_str(), b.length(), seed, hashB);
  hashA[0] ^= (0x3ac5 * hashB[0]);
  hashA[1] ^= (0x15db * hashB[1]);
  char buf[128];
  char* out = buf;
  for (int pos = 80; (pos -= 4) >= 0;) {
    int64_t nibble = ((pos >= 64) ? (hashA[1] >> (pos-64)) : (hashA[0] >> pos)) & 0xf;
    *out++ = ((nibble < 10) ? nibble+'0' : nibble-10+'a');
  }
  *out = 0;
  return hail::hstring(buf);
}

bool file_exists(const hail::hstring& name) {
  struct stat st;
  int rc = ::stat(name.c_str(), &st);
  return (rc == 0);
}

hail::hstring read_file_as_string(const hail::hstring& name) {
  FILE* f = fopen(name.c_str(), "r");
  if (!f) return hail::hstring("");
  hail::hstringstream ss;
  for (;;) {
    int c = fgetc(f);
    if (c == EOF) break;
    ss << (char)c;
  }
  fclose(f);
  return ss.str();
}

hail::hstring get_module_dir() {
  // This gives us a distinct temp directory for each process, so that
  // we can manage synchronization of threads accessing files in
  // the temp directory using only the in-memory big_mutex, rather than
  // dealing with the complexity of file locking.
  char buf[512];
  strcpy(buf, "/tmp/hail_XXXXXX");
  return ::mkdtemp(buf);
}

class ModuleConfig {
 public:
  bool is_darwin_;
  hail::hstring java_md_;
  hail::hstring ext_cpp_;
  hail::hstring ext_lib_;
  hail::hstring ext_mak_;
  hail::hstring module_dir_;

 public:
  ModuleConfig() :
#if defined(__APPLE__) && defined(__MACH__)
    is_darwin_(true),
#else
    is_darwin_(false),
#endif
    java_md_(is_darwin_ ? "darwin" : "linux"),
    ext_cpp_(".cpp"),
    ext_lib_(is_darwin_ ? ".dylib" : ".so"),
    ext_mak_(".mak"),
    module_dir_(get_module_dir()) {
  }
  
  hail::hstring get_lib_name(const hail::hstring& key) {
    hail::hstringstream ss;
    ss << module_dir_ << "/hm_" << key << ext_lib_;
    return ss.str();
  }
  
  void ensure_module_dir_exists() {
    int rc = ::access(module_dir_.c_str(), R_OK);
    if (rc < 0) { // create it
      rc = ::mkdir(module_dir_.c_str(), 0755);
      if (rc < 0) perror(module_dir_.c_str());
    }
  }
};

ModuleConfig config;

// module_table contains a single NativeModulePtr for each
// key that we have ever seen.  We never delete a NativeModule
// or unload its dynamically-loaded code.

std::map<hail::hstring, std::weak_ptr<NativeModule>> module_table;

NativeModulePtr module_find_or_make(
  const char* options,
  const char* source,
  const char* include
) {
  std::lock_guard<std::mutex> mylock(big_mutex);
  auto key = hash_two_strings(options, source);
  auto mod = module_table[key].lock();
  if (!mod) { // make it while holding the lock
    mod = std::make_shared<NativeModule>(options, source, include);
    module_table[key] = mod; // save a weak_ptr
  }
  return mod;
}

NativeModulePtr module_find_or_make(
  bool is_global,
  const char* key,
  ssize_t binary_size,
  const void* binary
) {
  std::lock_guard<std::mutex> mylock(big_mutex);
  auto mod = module_table[key].lock();
  if (!mod) { // make it while holding the lock
    mod = std::make_shared<NativeModule>(is_global, key, binary_size, binary);
    module_table[key] = mod; // save a weak_ptr
  }
  return mod;
}

} // end anon

// ModuleBuilder deals with compiling/linking source code to a DLL,
// and providing the binary DLL as an Array[Byte] which can be broadcast
// to all workers.

class ModuleBuilder {
 private:
  hail::hstring options_;
  hail::hstring source_;
  hail::hstring include_;
  hail::hstring key_;
  hail::hstring hm_base_;
  hail::hstring hm_mak_;
  hail::hstring hm_cpp_;
  hail::hstring hm_lib_;
  
public:
  ModuleBuilder(
    const hail::hstring& options,
    const hail::hstring& source,
    const hail::hstring& include,
    const hail::hstring& key
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
    fprintf(f, ".PHONY: FORCE\n");
    fprintf(f, "\n");
    fprintf(f, "MODULE    := hm_%s\n", key_.c_str());
    fprintf(f, "MODULE_SO := $(MODULE)%s\n", config.ext_lib_.c_str());
    fprintf(f, "ifndef JAVA_HOME\n");
    fprintf(f, "  TMP :=$(shell java -XshowSettings:properties -version 2>&1 | fgrep -i java.home)\n");
    fprintf(f, "  JAVA_HOME :=$(shell dirname $(filter-out java.home =,$(TMP)))\n");
    fprintf(f, "endif\n");
    fprintf(f, "JAVA_INCLUDE :=$(JAVA_HOME)/include\n");
    fprintf(f, "CXXFLAGS  := \\\n");
    fprintf(f, "  -std=c++11 -fPIC -march=native -fno-strict-aliasing -Wall \\\n");
    fprintf(f, "  -I$(JAVA_INCLUDE) \\\n");
    fprintf(f, "  -I$(JAVA_INCLUDE)/%s \\\n", config.java_md_.c_str());
    fprintf(f, "  -I%s \\\n", include_.c_str());
    bool have_oflag = (strstr(options_.c_str(), "-O") != nullptr);
    fprintf(f, "  %s%s \\\n", have_oflag ? "" : "-O3", options_.c_str());
    fprintf(f, "  -DHAIL_MODULE=$(MODULE)\n");
    fprintf(f, "LIBFLAGS := -fvisibility=default %s\n", 
      config.is_darwin_ ? "-dynamiclib -Wl,-undefined,dynamic_lookup"
                         : "-rdynamic -shared");
    fprintf(f, "\n");
    // build .so from .cpp
    fprintf(f, "$(MODULE_SO): FORCE\n");
    fprintf(f, "\t$(CXX) $(CXXFLAGS) -o $(MODULE).o -c $(MODULE).cpp 2> $(MODULE).err\n");
    fprintf(f, "\t$(CXX) $(CXXFLAGS) $(LIBFLAGS) -o $@ $(MODULE).o 2>> $(MODULE).err\n");
    fprintf(f, "\n");
    fclose(f);
  }

public:
  bool try_to_build() {
    write_mak();
    write_cpp();
    hail::hstringstream ss;
    ss << "make -B -C " << config.module_dir_ << " -f " << hm_mak_ << " 1>/dev/null";
    // run the command synchronously, while holding the big_mutex
    int rc = system(ss.str().c_str());
    if (rc != 0) {
      hail::hstring base(config.module_dir_ + "/hm_" + key_);
      fprintf(stderr, "makefile:\n%s", read_file_as_string(base+".mak").c_str());
      fprintf(stderr, "errors:\n%s",   read_file_as_string(base+".err").c_str());
    }
    return (rc == 0);
  }
};

NativeModule::NativeModule(
  const char* options,
  const char* source,
  const char* include
) :
  build_state_(kInit),
  load_state_(kInit),
  key_(hash_two_strings(options, source)),
  is_global_(false),
  dlopen_handle_(nullptr),
  lib_name_(config.get_lib_name(key_)) {
  // Master constructor - try to get module built in local file
  config.ensure_module_dir_exists();
  if (file_exists(lib_name_)) {
    build_state_ = kPass;
  } else {
    // The file doesn't exist, let's build it
    ModuleBuilder builder(options, source, include, key_);
    build_state_ = (builder.try_to_build() ? kPass : kFail);
  }
}

NativeModule::NativeModule(
  bool is_global,
  const char* key,
  ssize_t binary_size,
  const void* binary
) :
  build_state_(is_global ? kPass : kInit),
  load_state_(is_global ? kPass : kInit),
  key_(key),
  is_global_(is_global),
  dlopen_handle_(nullptr),
  lib_name_(config.get_lib_name(key_)) {
  // Worker constructor - try to get the binary written to local file
  if (is_global_) return;
  int rc = 0;
  config.ensure_module_dir_exists();
  build_state_ = kPass; // unless we get an error after this
  if (!file_exists(lib_name_)) {
    // We hold big_mutex so there is no race
    if (binary_size == 0) {
      // This binary came from a lib which gave errors during reading
      build_state_ = kFail;
    } else {
      int fd = open(lib_name_.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
      if (fd < 0) {
        build_state_ = kFail;
      } else {
        rc = write(fd, binary, binary_size);
        if (rc != binary_size) {
          build_state_ = kFail;
        }
        ::close(fd);
      }
    }
  }
  if (build_state_ == kPass) try_load_locked();
}

NativeModule::~NativeModule() {
  if (!is_global_ && dlopen_handle_) {
    dlclose(dlopen_handle_);
  }
}

bool NativeModule::try_load_locked() {
  if (load_state_ == kInit) {
    assert(!is_global_);
    auto handle = dlopen(lib_name_.c_str(), RTLD_GLOBAL|RTLD_NOW);
    if (handle) {
      dlopen_handle_ = handle;
      load_state_ = kPass;
    } else {
      // Attempts to find a func will give a bad NativeStatus
      load_state_ = kFail;
    }
  }
  return (load_state_ == kPass);
}

std::vector<char> NativeModule::get_binary() {
  std::lock_guard<std::mutex> mylock(big_mutex);
  std::vector<char> empty;
  if (build_state_ == kFail) {
    return empty;
  }
  int fd = open(config.get_lib_name(key_).c_str(), O_RDONLY, 0666);
  if (fd < 0) {
    return empty; // build failed, no lib, return empty
  }
  struct stat st;
  int rc = fstat(fd, &st);
  if (rc < 0) {
    return empty;
  }
  std::vector<char> vec;
  size_t file_size = st.st_size;
  vec.resize(file_size);
  rc = read(fd, &vec[0], file_size);
  if ((size_t)rc != file_size) {
    return empty;
  }
  close(fd);
  return vec;
}

static hail::hstring to_qualified_name(
  JNIEnv* env,
  const hail::hstring& key,
  jstring nameJ,
  int numArgs,
  bool is_global,
  bool is_longfunc
) {
  JString name(env, nameJ);
  hail::hstring result;
  if (is_global) {
    // No name-mangling for global func names
    result = name;
  } else {
    // Mangled name for hail::hm_<key>::funcname(NativeStatus* st, some number of longs)
    hail::hstringstream ss;
    auto mod_name = hail::hstring("hm_") + key;
    ss << "_ZN4hail" 
       << (uint64_t)mod_name.length() << mod_name
       << (uint64_t)strlen(name) << (const char*)name
       << "E"
       << "P12NativeStatus";
    for (int j = 0; j < numArgs; ++j) ss << 'l';
    result = ss.str();
  }
  return result;
}

void NativeModule::find_LongFuncL(
  JNIEnv* env,
  NativeStatus* st,
  jobject funcObj,
  jstring nameJ,
  int numArgs
) {
  std::lock_guard<std::mutex> mylock(big_mutex);
  void* funcAddr = nullptr;
  if (!try_load_locked()) {
    NATIVE_ERROR(st, 1001, "ErrModuleNotFound");
  } else {
    auto qualName = to_qualified_name(env, key_, nameJ, numArgs, is_global_, true);
    funcAddr = ::dlsym(is_global_ ? RTLD_DEFAULT : dlopen_handle_, qualName.c_str());
    if (!funcAddr) {
      fprintf(stderr, "ErrLongFuncNotFound \"%s\"\n", qualName.c_str());
      NATIVE_ERROR(st, 1003, "ErrLongFuncNotFound dlsym(\"%s\")", qualName.c_str());
    }
  }
  NativeObjPtr ptr = std::make_shared< NativeFuncObj<long> >(shared_from_this(), funcAddr);
  init_NativePtr(env, funcObj, &ptr);
}

void NativeModule::find_PtrFuncL(
  JNIEnv* env,
  NativeStatus* st,
  jobject funcObj,
  jstring nameJ,
  int numArgs
) {
  std::lock_guard<std::mutex> mylock(big_mutex);
  void* funcAddr = nullptr;
  if (!try_load_locked()) {
    NATIVE_ERROR(st, 1001, "ErrModuleNotFound");
  } else {
    auto qualName = to_qualified_name(env, key_, nameJ, numArgs, is_global_, false);
    funcAddr = ::dlsym(is_global_ ? RTLD_DEFAULT : dlopen_handle_, qualName.c_str());
    if (!funcAddr) {
      fprintf(stderr, "ErrPtrFuncNotFound \"%s\"\n", qualName.c_str());
      NATIVE_ERROR(st, 1003, "ErrPtrFuncNotFound dlsym(\"%s\")", qualName.c_str());
    }
  }
  NativeObjPtr ptr = std::make_shared< NativeFuncObj<NativeObjPtr> >(shared_from_this(), funcAddr);
  init_NativePtr(env, funcObj, &ptr);
}

// Functions implementing NativeModule native methods

static NativeModule* to_NativeModule(JNIEnv* env, jobject obj) {
  return static_cast<NativeModule*>(get_from_NativePtr(env, obj));
}

NATIVEMETHOD(void, NativeModule, nativeCtorMaster)(
  JNIEnv* env,
  jobject thisJ,
  jstring optionsJ,
  jstring sourceJ,
  jstring includeJ
) {
  JString options(env, optionsJ);
  JString source(env, sourceJ);
  JString include(env, includeJ);
  NativeObjPtr ptr = module_find_or_make(options, source, include);
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
  ssize_t binary_size = env->GetArrayLength(binaryJ);
  auto binary = env->GetByteArrayElements(binaryJ, 0);
  NativeObjPtr ptr = module_find_or_make(is_global, key, binary_size, binary);
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
  if (mod->build_state_ != NativeModule::kPass) {
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
  auto vec = mod->get_binary();
  jbyteArray result = env->NewByteArray(vec.size());
  jbyte* rbuf = env->GetByteArrayElements(result, 0);
  memcpy(rbuf, &vec[0], vec.size());
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
