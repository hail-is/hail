#include "hail/NativeMethod.h"
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
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iterator>

namespace hail {

namespace {

// Top-level NativeModule methods lock this mutex.  Constructors, and helper methods
// with names ending in "_locked", must be called only while holding the mutex.
//
// That makes everything single-threaded.

std::mutex big_mutex;

// A simple way to get a hash of two strings, take 80bits,
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

bool file_exists(const std::string& name) {
  struct stat st;
  int rc = ::stat(name.c_str(), &st);
  return (rc == 0);
}

std::string get_module_dir() {
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
  constexpr static bool is_darwin_{
#if defined(__APPLE__) && defined(__MACH__)
    true
#else
    false
#endif
  };
  const char *const cxx_{"c++"};
  std::string module_dir_;
  std::string java_home_{""};

  const char* java_md_{is_darwin_ ? "darwin" : "linux"};
  const char* ext_lib_{is_darwin_ ? ".dylib" : ".so"};
  std::array<const char*, 8> cxx_flags_ {
    "-x",
    "c++",
    "-std=c++11",
    "-pipe",
    "-march=native",
    "-fPIC",
    "-fno-strict-aliasing",
    "-Wall",
  };
  std::array<const char*, 3> ld_flags_ {
      "-fvisibility=default",
      is_darwin_ ? "-dynamiclib" : "-rdynamic",
      is_darwin_ ? "-Wl,-undefined,dynamic_lookup" : "-shared",
  };
  std::array<const char*, 1> lib_flags_{"-llz4"};

 public:
  ModuleConfig() :
    module_dir_(get_module_dir()) {
    const char* tmp = getenv("JAVA_HOME");
    if (tmp != nullptr) {
      java_home_ = tmp;
    } else {
      auto cmd = "java -XshowSettings:properties -version 2>&1 "
          "| sed -n 's:\\s\\+java.home =\\s\\+\\(/.*\\)$:\\1:gp' "
          "| xargs dirname | tr -d '\\n'";
      auto p = popen(cmd, "r");
      if (!p) { perror("popen"); return; }
      int c;
      while ((c = fgetc(p)) != EOF) {
        java_home_.push_back(static_cast<char>(c));
      }
      int e = pclose(p);
      if (e != 0) {
        perror("get java home");
      }
    }
  }

  std::string get_lib_name(const std::string& key) {
    std:: stringstream ss;
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

std::unordered_map<std::string, std::weak_ptr<NativeModule>> module_table;

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
  std::string source_;
  std::string include_;
  std::string key_;
  std::string hm_base_;
  std::string hm_lib_;
  std::vector<std::string> options_;

public:
  ModuleBuilder(
    const std::string& options,
    const std::string& source,
    const std::string& include,
    const std::string& key
  ) :
    source_(source),
    include_(include),
    key_(key) {
    // To start with, put dynamic code in $HOME/hail_modules
    std::istringstream iss{options};
    options_ = std::vector<std::string>{std::istream_iterator<std::string>(iss),
                                        std::istream_iterator<std::string>()};
    auto base = (config.module_dir_ + "/hm_") + key_;
    hm_base_ = base;
    hm_lib_ = (base + config.ext_lib_);
  }

  virtual ~ModuleBuilder() { }

public:
  bool try_to_build() {
    // create the command invocation
    std::array<const char*, 2> file_params{"-", "-o"};
    std::vector<const char*> cmd{config.cxx_};
    std::vector<std::string> include_args;
    std::string module_define{"-DHAIL_MODULE=hm_" + key_};
    include_args.push_back("-I" + include_);
    include_args.push_back("-I" + config.java_home_ + "/include");
    include_args.push_back(include_args.back() + "/" + config.java_md_);

    cmd.insert(cmd.end(), config.cxx_flags_.begin(), config.cxx_flags_.end());
    for (auto& s : include_args) cmd.push_back(s.c_str());
    for (auto& s : options_) cmd.push_back(s.c_str());
    cmd.push_back(module_define.c_str());
    cmd.insert(cmd.end(), config.ld_flags_.begin(), config.ld_flags_.end());
    cmd.insert(cmd.end(), file_params.begin(), file_params.end());
    cmd.push_back(hm_lib_.c_str());
    cmd.insert(cmd.end(), config.lib_flags_.begin(), config.lib_flags_.end());
    cmd.push_back(nullptr);

    int pipefd[2];
    if (pipe(pipefd) < 0) return perror("c++ compile, pipe"), false;
    int read_src_fd = pipefd[0];
    int write_src_fd = pipefd[1];
    if (pipe(pipefd) < 0) return perror("c++ compile, pipe"), false;
    int read_err_fd = pipefd[0];
    int write_err_fd = pipefd[1];

    pid_t pid = fork();

    if (pid < 0) {
      perror("c++ compile, fork");
      close(write_src_fd);
      close(read_src_fd);
      close(write_err_fd);
      close(read_err_fd);

      return false;
    }

    if (pid == 0) { // child
      close(write_src_fd);
      close(read_err_fd);
      if (dup2(read_src_fd, STDIN_FILENO) < 0) {
        perror("c++ compile, child, dup2");
        exit(EXIT_FAILURE);
      }

      if (dup2(write_err_fd, STDERR_FILENO) < 0) {
        perror("c++ compile, child, dup2");
        exit(EXIT_FAILURE);
      }

      execvp(cmd[0], const_cast<char* const*>(cmd.data()));
      perror("c++ compile, child, execve");
      exit(EXIT_FAILURE);
    }

    // parent
    bool st = true;
    int compile_st = 0;
    close(read_src_fd);
    close(write_err_fd);
    size_t nread = 0;
    ssize_t r;
    while ((r = write(write_src_fd, source_.data() + nread, source_.length() - nread)) > 0
            && nread < source_.length()) {
      nread += r;
    }

    if (r < 0 && errno != EPIPE) {
      st = false;
      perror("c++ compile, parent, write");
    }
    close(write_src_fd);

    std::string err;
    std::array<char, BUFSIZ> buf;
    while ((r = read(read_err_fd, buf.data(), sizeof(buf))) > 0)
      err.insert(err.end(), buf.begin(), buf.begin() + r);
    close(read_err_fd);

    waitpid(pid, &compile_st, 0);
    if ((WIFEXITED(compile_st) && WEXITSTATUS(compile_st) != EXIT_SUCCESS) || WIFSIGNALED(compile_st)) {
      fprintf(stderr, "c++ compile, compiler exited with errors:\n");
      fprintf(stderr, "command:\n");
      auto first = true;
      for (auto& s : cmd) {
        fprintf(stderr, "%s%s \\\n", first ? "" : "\t", s);
        first = false;
      }
      fprintf(stderr, "errors:\n%s\n", err.c_str());

      auto cpp_name = hm_base_ + ".cpp";
      FILE* f = fopen(cpp_name.c_str(), "w");
      if (!f) {
        fprintf(stderr, "could not open c++ file for dumping invalid code, printing to stdout instead:\n");
        fprintf(stderr, "%s\n", source_.c_str());
      } else {
        fwrite(source_.data(), 1, source_.length(), f);
        fclose(f);
        fprintf(stderr, "invalid c++ code at: %s\n", cpp_name.c_str());
      }
      return false;
    }

    return st;
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
      fprintf(stderr, "dlopen: %s\n", dlerror());
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

static std::string to_qualified_name(
  JNIEnv* env,
  const std::string& key,
  jstring nameJ,
  int numArgs,
  bool is_global,
  bool
) {
  JString name(env, nameJ);
  std::string result;
  if (is_global) {
    // No name-mangling for global func names
    result = name;
  } else {
    // Mangled name for hail::hm_<key>::funcname(NativeStatus* st, some number of longs)
    std::stringstream ss;
    auto mod_name = std::string("hm_") + key;
    ss << "_ZN4hail"
       << mod_name.length() << mod_name
       << strlen(name) << (const char*)name
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
