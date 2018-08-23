#include "hail/NativeModule.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
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
#include <map>
#include <memory>
#include <mutex>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace hail {

namespace {

std::mutex big_mutex;

// File-polling interval in usecs
const int kFilePollMicrosecs = 50000;

// Timeout for compile/link of a DLL
const int kBuildTimeoutSecs = 300;

// A quick-and-dirty way to get a hash of two strings, take 80bits,
// and produce a 20byte string of hex digits.  We also sprinkle
// in some "salt" from a checksum of a tar of all header files, so
// that any change to header files will force recompilation.
//
// The shorter string (corresponding to options), may have only a
// few distinct values, so we need to mix it up with the longer
// string in various ways.

std::string even_bytes(const std::string& a) {
  std::stringstream ss;
  size_t len = a.length();
  for (size_t j = 0; j < len; j += 2) {
    ss << a[j];
  }
  return ss.str();
}

std::string hash_two_strings(const std::string& a, const std::string& b) {
  bool a_shorter = (a.length() < b.length());
  const std::string* shorter = (a_shorter ? &a : &b);
  const std::string* longer  = (a_shorter ? &b : &a);
  uint64_t hashA = std::hash<std::string>()(*longer);
  uint64_t hashB = std::hash<std::string>()(*shorter + even_bytes(*longer));
  if (sizeof(size_t) < 8) {
    // On a 32bit machine we need to work harder to get 80 bits
    uint64_t hashC = std::hash<std::string>()(*longer + "SmallChangeForThirdHash");
    hashA += (hashC << 32);
  }
  if (a_shorter) hashA ^= 0xff; // order of strings should change result
  hashA ^= ALL_HEADER_CKSUM; // checksum from all header files
  hashA ^= (0x3ac5*hashB); // mix low bits of hashB into hashA
  hashB &= 0xffff;
  char buf[128];
  char* out = buf;
  for (int pos = 80; (pos -= 4) >= 0;) {
    int64_t nibble = ((pos >= 64) ? (hashB >> (pos-64)) : (hashA >> pos)) & 0xf;
    *out++ = ((nibble < 10) ? nibble+'0' : nibble-10+'a');
  }
  *out = 0;
  return std::string(buf);
}

bool file_stat(const std::string& name, struct stat* st) {
  // Open file for reading to avoid inconsistent cached attributes
  int fd = ::open(name.c_str(), O_RDONLY, 0666);
  if (fd < 0) return false;
  int rc;
  do {
    errno = 0;
    rc = ::fstat(fd, st);
  } while ((rc < 0) && (errno == EINTR));
  ::close(fd);
  return (rc == 0);
}

bool file_exists_and_is_recent(const std::string& name) {
  time_t now = ::time(nullptr);
  struct stat st;
  return (file_stat(name, &st) && (st.st_ctime+kBuildTimeoutSecs > now));
}

bool file_exists(const std::string& name) {
  struct stat st;
  return file_stat(name, &st);
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

std::string strip_suffix(const std::string& s, const char* suffix) {
  size_t len = s.length();
  size_t n = strlen(suffix);
  if ((n > len) || (strncmp(&s[len-n], suffix, n) != 0)) return s;
  return std::string(s, 0, len-n);
}

std::string get_cxx_name() {
  char* p = ::getenv("CXX");
  if (p) return std::string(p);
  auto s = run_shell_get_first_line("which c++");
  if (strstr(s.c_str(), "c++")) return s;
  // The last guess is to just say "c++"
  return std::string("c++");
}

std::string get_cxx_std(const std::string& cxx) {
  const char* standards[] = { "-std=c++17", "-std=c++14" };
  int fd = ::open("/tmp/.hail_empty.cc", O_WRONLY|O_CREAT|O_TRUNC, 0666);
  if (fd >= 0) ::close(fd);
  for (int j = 0; j < 2; ++j) {
    std::stringstream ss;
    ss << cxx << " " << standards[j] << " /tmp/.hail_empty.cc 2>1";
    auto s = run_shell_get_first_line(ss.str().c_str());
    if (!strstr(s.c_str(), "unrecognized") && !strstr(s.c_str(), "invalid")) {
      return standards[j];
    }
  }
  // Default to pass "-std=c++11", a minimum requirement
  return "-std=c++11";
}

std::string get_module_dir() {
  // This gives us a distinct temp directory for each process, so that
  // we can manage synchronization between different of the files in the
  // temp directory using only in-memory std::mutex'es, rather than
  // dealing with complexities of file locking.
  char buf[512];
  strcpy(buf, "/tmp/hail_XXXXXX");
  return ::mkdtemp(buf);
}

std::string get_perl_name() {
  std::string name("/usr/bin/perl");
  if (file_exists(name)) return name;
  name = run_shell_get_first_line("which perl 2>/dev/null");
  if (strstr(name.c_str(), "perl")) return name;
  // The last guess is to just say "perl"
  return std::string("perl");
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
  std::string cxx_std_;
  std::string java_md_;
  std::string perl_name_;

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
    module_dir_(get_module_dir()),
    cxx_name_(get_cxx_name()),
    cxx_std_(get_cxx_std(cxx_name_)),
    java_md_(is_darwin_ ? "darwin" : "linux"),
    perl_name_(get_perl_name()) {
  }
  
  std::string get_lock_name(const std::string& key) {
    std::stringstream ss;
    ss << module_dir_ << "/hm_" << key << ".lock";
    return ss.str();
  }
  
  std::string get_lib_name(const std::string& key) {
    std:: stringstream ss;
    ss << module_dir_ << "/hm_" << key << ext_lib_;
    return ss.str();
  }
  
  std::string get_new_name(const std::string& key) {
    std:: stringstream ss;
    ss << module_dir_ << "/hm_" << key << ext_new_;
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

// module_table is used to ensure that several Spark workers will get
// shared_ptr's to a single NativeModule, rather than each having their
// own NativeModule.  Callers must hold the big_mutex.

std::unordered_map<std::string, std::weak_ptr<NativeModule>> module_table;

} // end anon

// ModuleBuilder deals with compiling/linking source code to a DLL,
// and providing the binary DLL as an Array[Byte] which can be broadcast
// to all workers.

class ModuleBuilder {
private:
  NativeModule* parent_;
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
    NativeModule* parent,
    const std::string& options,
    const std::string& source,
    const std::string& include,
    const std::string& key
  ) :
    parent_(parent),
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
    // When we replace foo.new, or foo.{so,dylib}, that must be done
    // with an atomic-rename operation which guarantees not to leave any
    // window when there is no foo.new.  On Linux, "mv -f" is atomic, but
    // on MacOS it isn't.  After some experimentation, we now use perl's
    // rename command, with the belief that on POSIX-compatible systems
    // this will be implemented as a rename() system-call, specified thus:
    //
    // "If the link named by the new argument exists, it shall be removed and 
    //  old renamed to new. In this case, a link named new shall remain visible
    //  to other processes throughout the renaming operation and refer either
    //  to the file referred to by new or old before the operation began."
    //
    // Since perl has a nasty habit of doing funny things to some characters
    // in non-quoted strings, we take care to quote the filenames.
    //
    // It's enormously confusing to be writing C++ to write a makefile
    // which contains embedded bash scripts which call perl.  That was not
    // the original design, but issues of MacOS-vs-Linux portability left
    // only a narrow path to a working solution.
    FILE* f = fopen(hm_mak_.c_str(), "w");
    if (!f) { perror("fopen"); return; }
    std::string javaHome = config.java_home_;
    std::string javaMD = config.java_md_;
    fprintf(f, "MODULE    := hm_%s\n", key_.c_str());
    fprintf(f, "MODULE_SO := $(MODULE)%s\n", config.ext_lib_.c_str());
    fprintf(f, "PERL      := %s\n", config.perl_name_.c_str());
    fprintf(f, "ifndef JAVA_HOME\n");
    fprintf(f, "  TMP :=$(shell java -XshowSettings:properties -version 2>&1 | fgrep -i java.home)\n");
    fprintf(f, "  JAVA_HOME :=$(shell dirname $(filter-out java.home =,$(TMP)))\n");
    fprintf(f, "endif\n)
    fprintf(f, "CXX       := %s\n", config.cxx_name_.c_str());
    fprintf(f, "CXXFLAGS  := \\\n");
    fprintf(f, "  %s -fPIC -march=native -fno-strict-aliasing -Wall \\\n", config.cxx_std_.c_str());
    auto java_include = strip_suffix(config.java_home_, "/jre") + "/include";
    fprintf(f, "  -I%s \\\n", java_include.c_str());
    fprintf(f, "  -I%s/%s \\\n", java_include.c_str(), config.java_md_.c_str());
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
    fprintf(f, "\t$(PERL) -e 'rename \"$(MODULE).new\", \"$@\"'\n");
    fprintf(f, "\n");
    // build .o from .cpp
    fprintf(f, "$(MODULE).o: $(MODULE).cpp\n");
    fprintf(f, "\t$(CXX) $(CXXFLAGS) -o $@ -c $< 2> $(MODULE).err && \\\n");
    fprintf(f, "\t  $(CXX) $(CXXFLAGS) $(LIBFLAGS) -o $(MODULE).tmp $(MODULE).o 2>> $(MODULE).err ; \\\n");
    fprintf(f, "\tstatus=$$? ; \\\n");
    fprintf(f, "\tif [ $$status -ne 0 ] || [ -z $(MODULE).tmp ]; then \\\n");
    fprintf(f, "\t  /bin/rm -f $(MODULE).new ; \\\n");
    fprintf(f, "\t  echo FAIL ; exit 1 ; \\\n");
    fprintf(f, "\tfi\n");
    fprintf(f, "\t-/bin/rm -f $(MODULE).err\n");
    fprintf(f, "\t$(PERL) -e 'rename \"$(MODULE).tmp\", \"$(MODULE).new\"'\n");
    fprintf(f, "\n");
    fclose(f);
  }

public:
  bool try_to_start_build() {
    // Try to create the .new file, we hold the global lock so there is no race
    int fd = ::open(hm_new_.c_str(), O_WRONLY|O_CREAT, 0666);
    if (fd < 0) {
      perror("open");
      assert(false);
    }
    ::close(fd);
    // The .new file may look the same age as the .cpp file, but
    // the makefile is written to ignore the .new timestamp
    write_mak();
    write_cpp();
    std::stringstream ss;
    ss << "/usr/bin/make -B -C " << config.module_dir_ << " -f " << hm_mak_;
    ss << " 1>/dev/null &";
    int rc = system(ss.str().c_str());
    if (rc == -1) {
      fprintf(stderr, "DEBUG: system() -> -1\n");
      perror("system");
      ::unlink(hm_new_.c_str()); // the build is not running, so recover ASAP
    }
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
  lock_name_(config.get_lock_name(key_)),
  lib_name_(config.get_lib_name(key_)),
  new_name_(config.get_new_name(key_)) {
  // Master constructor - try to get module built in local file
  config.ensure_module_dir_exists();
  lock();
  bool have_lib = (!force_build && file_exists(lib_name_));
  unlock();
  if (have_lib) {
    build_state_ = kPass;
  } else {
    // The file doesn't exist, let's start building it
    ModuleBuilder builder(this, options, source, include, key_);
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
  lock_name_(config.get_lock_name(key_)),
  lib_name_(config.get_lib_name(key_)),
  new_name_(config.get_new_name(key_)) {
  // Worker constructor - try to get the binary written to local file
  if (is_global_) return;
  int rc = 0;
  config.ensure_module_dir_exists();
  for (;;) {
    struct stat lib_stat;
    if (file_stat(lib_name_, &lib_stat) && (lib_stat.st_size == binary_size)) {
      build_state_ = kPass;
      break;
    }
    // Race to write the new file
    int fd = open(new_name_.c_str(), O_WRONLY|O_CREAT|O_EXCL, 0666);
    if (fd >= 0) {
      // Now we're about to write the new file
      rc = write(fd, binary, binary_size);
      assert(rc == binary_size);
      ::close(fd);
      ::chmod(new_name_.c_str(), 0644);
      struct stat st;
      if (file_stat(lib_name_, &st)) {
        long old_size = st.st_size;
        if (old_size == binary_size) {
          ::unlink(new_name_.c_str());
          break;
        } else if (old_size == 0) {
          ::unlink(lib_name_.c_str());
        } else {
          auto old = lib_name_ + ".old";
          ::rename(lib_name_.c_str(), old.c_str());
        }
      }
      // Don't let anyone see the file until it is completely written
      rc = ::rename(new_name_.c_str(), lib_name_.c_str());
      build_state_ = ((rc == 0) ? kPass : kFail);
      break;
    } else {
      // Someone else is writing to new
      while (file_exists_and_is_recent(new_name_)) {
        usleep_without_lock(kFilePollMicrosecs);
      }
    }
  }
  if (build_state_ == kPass) {
    try_load();
  }
}

NativeModule::~NativeModule() {
  if (!is_global_ && dlopen_handle_) {
    dlclose(dlopen_handle_);
  }
}

void NativeModule::usleep_without_lock(int64_t usecs) {
  big_mutex.unlock();
  usleep(usecs);
  big_mutex.lock();
}

bool NativeModule::try_wait_for_build() {
  if (build_state_ == kInit) {
    // The writer will rename new to lib.  If we tested exists(lib)
    // followed by exists(new) then the rename could occur between
    // the two tests. This way is safe provided that either rename is atomic,
    // or rename creates the new name before destroying the old name.
    while (file_exists_and_is_recent(new_name_)) {
      usleep_without_lock(kFilePollMicrosecs);
    }
    struct stat st;
    if (file_stat(new_name_, &st) && (st.st_ctime+kBuildTimeoutSecs < time(nullptr))) {
      fprintf(stderr, "WARNING: force break new %s\n", new_name_.c_str());
      ::unlink(new_name_.c_str()); // timeout
    }
    build_state_ = (file_exists(lib_name_) ? kPass : kFail);
    if (build_state_ == kFail) {
      std::string base(config.module_dir_ + "/hm_" + key_);
      fprintf(stderr, "makefile:\n%s", read_file_as_string(base+".mak").c_str());
      fprintf(stderr, "errors:\n%s",   read_file_as_string(base+".err").c_str());
    }
  }
  return (build_state_ == kPass);
}

bool NativeModule::try_load() {
  if (load_state_ == kInit) {
    if (is_global_) {
      load_state_ = kPass;
    } else if (!try_wait_for_build()) {
      load_state_ = kFail;
    } else {
      // At first this had no mutex and RTLD_LAZY, but MacOS tests crashed
      // when two threads loaded the same .dylib.
      auto handle = dlopen(lib_name_.c_str(), RTLD_GLOBAL|RTLD_NOW);
      if (!handle) {
        fprintf(stderr, "ERROR: dlopen failed: %s\n", dlerror());
      }
      load_state_ = (handle ? kPass : kFail);
      if (handle) dlopen_handle_ = handle;
    }
  }
  return (load_state_ == kPass);
}

static std::string to_qualified_name(
  JNIEnv* env,
  const std::string& key,
  jstring nameJ,
  int numArgs,
  bool is_global,
  bool is_longfunc
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
    sprintf(buf, "_ZN4hail%lu%s%lu%sE%s%s",
      moduleName.length(), moduleName.c_str(), strlen(name), (const char*)name, 
      "P12NativeStatus", argTypeCodes);
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
  void* funcAddr = nullptr;
  if (!try_load()) {
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
  jstring includeJ,
  jboolean force_buildJ
) {
  JString options(env, optionsJ);
  JString source(env, sourceJ);
  JString include(env, includeJ);
  bool force_build = (force_buildJ != JNI_FALSE);
  std::lock_guard<std::mutex> mylock(big_mutex);
  NativeObjPtr ptr = std::make_shared<NativeModule>(options, source, include, force_build);
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
  std::lock_guard<std::mutex> mylock(big_mutex);
  NativeModulePtr mod;
  auto iter = module_cache[std::string(key)];
  if (iter != module_cache.end()) mod = iter->second.lock();
  if (!mod) {
    long binary_size = env->GetArrayLength(binaryJ);
    auto binary = env->GetByteArrayElements(binaryJ, 0);
    mod = std::make_shared<NativeModule>(is_global, key, binary_size, binary);
    module_cache.insert(mod);
    env->ReleaseByteArrayElements(binaryJ, binary, JNI_ABORT);
  }
  NativeObjPtr ptr = mod;
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
  std::lock_guard<std::mutex> mylock(big_mutex);
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
  std::lock_guard<std::mutex> mylock(big_mutex);
  mod->try_wait_for_build();
  int fd = open(config.get_lib_name(mod->key_).c_str(), O_RDONLY, 0666);
  if (fd < 0) {
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
  std::lock_guard<std::mutex> mylock(big_mutex); \
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
