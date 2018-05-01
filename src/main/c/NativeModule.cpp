//
// src/main/c/NativeModule.cpp - native funcs for Scala NativeModule
//
// Richard Cownie, Hail Team, 2018-04-12
//
#include "hail/NativeModule.h"
#include "hail/CommonDefs.h"
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

NAMESPACE_BEGIN(hail)

NAMESPACE_BEGIN_ANON

//
// File-polling interval in usecs
//
const int kFilePollMicrosecs = 50000;

//
// A quick-and-dirty way to get a hash of two strings, take 80bits,
// and produce a 20byte string of hex digits.
//
std::string hashTwoStrings(const std::string& a, const std::string& b) {
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

bool fileExistsAndIsRecent(const std::string& name) {
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

bool fileExists(const std::string& name) {
  int rc;
  struct stat st;
  do {
    errno = 0;
    rc = stat(name.c_str(), &st);
  } while ((rc < 0) && (errno == EINTR));
  return(rc == 0);
}

long fileSize(const std::string& name) {
  int rc;
  struct stat st;
  do {
    errno = 0;
    rc = ::stat(name.c_str(), &st);
  } while ((rc < 0) && (errno = EINTR));
  return((rc < 0) ? -1 : st.st_size);
}

std::string readFileAsString(const std::string& name) {
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

const char* getenvWithDefault(const char* name, const char* defaultVal) {
  const char* s = ::getenv(name);
  return(s ? s : defaultVal);
}

std::string lastMatch(const char* pattern) {
  std::stringstream ss;
  ss << "/bin/ls -d " << pattern << " 2>/dev/null | tail -1";
  FILE* f = popen(ss.str().c_str(), "r");
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
  //fprintf(stderr, "DEBUG: lastMatch(%s) -> %s\n", pattern, buf);
  return std::string(buf);
}

class ModuleConfig {
public:
  bool isOsDarwin_;
  std::string extCpp_;
  std::string extLib_;
  std::string extMak_;
  std::string extNew_;
  std::string moduleDir_;
  std::string cxxName_;
  std::string llvmHome_;
  std::string javaHome_;
  std::string javaMD_;

public:
  ModuleConfig() :
#if defined(__APPLE__) && defined(__MACH__)
    isOsDarwin_(true),
#else
    isOsDarwin_(false),
#endif
    extCpp_(".cpp"),
    extLib_(isOsDarwin_ ? ".dylib" : ".so"),
    extMak_(".mak"),
    extNew_(".new"),
    moduleDir_() {
    const char* envHome = getenvWithDefault("HOME", "/tmp");
    moduleDir_ = (std::string(envHome) + "/hail_modules");
    if (isOsDarwin_) {
      llvmHome_ = lastMatch("/usr /usr/local/*llvm-6*x86_64*darwin");
      cxxName_ = (fileExists(llvmHome_+"/bin/clang") ? "clang" : "c++");
      if (cxxName_ == "c++") llvmHome_ = "/usr";
      javaHome_ = "/Library/Java/JavaVirtualMachines/jdk1.8.0_162.jdk/Contents/Home";
      javaMD_ = "darwin";
    } else {
      llvmHome_ = lastMatch("/usr/l*/llvm* /usr/l*/llvm-5* /usr/l*/llvm-6*");
      cxxName_ = (fileExists(llvmHome_+"/bin/clang") ? "clang" : "g++");
      javaHome_ = "/usr/lib/jvm/default-java";
      javaMD_ = "linux";
    }
  }
  
  std::string getLibName(const std::string& key) {
    std:: stringstream ss;
    ss << moduleDir_ << "/hm_" << key  << extLib_;
    return ss.str();
  }
  
  std::string getNewName(const std::string& key) {
    std:: stringstream ss;
    ss << moduleDir_ << "/hm_" << key  << extNew_;
    return ss.str();
  }
  
  void ensureModuleDirExists() {
    int rc = ::access(moduleDir_.c_str(), R_OK);
    if (rc < 0) { // create it
      rc = ::mkdir(moduleDir_.c_str(), 0666);
      if (rc < 0) perror(moduleDir_.c_str());
      rc = ::chmod(moduleDir_.c_str(), 0755);
    }
  }
};

ModuleConfig config;

NAMESPACE_END_ANON

//
// ModuleBuilder deals with compiling/linking source code to a DLL,
// and providing the binary DLL as an Array[Byte] which can be broadcast
// to all workers.
//

class ModuleBuilder {
private:
  std::string options_;
  std::string source_;
  std::string include_;
  std::string key_;
  std::string moduleBase_;
  std::string moduleMak_;
  std::string moduleCpp_;
  std::string moduleNew_;
  std::string moduleLib_;
  
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
    auto base = (config.moduleDir_ + "/hm_") + key_;
    moduleBase_ = base;
    moduleMak_ = (base + config.extMak_);
    moduleCpp_ = (base + config.extCpp_);
    moduleNew_ = (base + config.extNew_);
    moduleLib_ = (base + config.extLib_);
  }
  
  virtual ~ModuleBuilder() { }
  
private:
  void writeSource() {
    FILE* f = fopen(moduleCpp_.c_str(), "w");
    if (!f) { perror("fopen"); return; }
    fwrite(source_.data(), 1, source_.length(), f);
    fclose(f);
  }
  
  void writeMakefile() {
    FILE* f = fopen(moduleMak_.c_str(), "w");
    if (!f) { perror("fopen"); return; }
    std::string javaHome = config.javaHome_;
    std::string javaMD = config.javaMD_;
    fprintf(f, "MODULE    := hm_%s\n", key_.c_str());
    fprintf(f, "MODULE_SO := $(MODULE)%s\n", config.extLib_.c_str());
    fprintf(f, "CXX       := %s/bin/%s\n", 
      config.llvmHome_.c_str(), config.cxxName_.c_str());
    // Downgrading from -std=c++14 to -std=c++11 for CI w/ old compilers
    fprintf(f, "CXXFLAGS  := \\\n");
    fprintf(f, "  -std=c++11 -fPIC -march=native -fno-strict-aliasing -Wall -Werror \\\n");
    fprintf(f, "  -I%s/include -I%s/include/%s \\\n", 
      javaHome.c_str(), javaHome.c_str(), javaMD.c_str());
    const char* userOptions = options_.c_str();
    fprintf(f, "  %s%s\\\n",
      strstr(userOptions, "-O") ? "" : "-O3 ", userOptions);
    fprintf(f, "  -I%s \\\n", include_.c_str());
    fprintf(f, "  -DHAIL_MODULE=$(MODULE)\n");
    fprintf(f, "LIBFLAGS := -fvisibility=default %s\n", 
      config.isOsDarwin_ ? "-dynamiclib -Wl,-undefined,dynamic_lookup"
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
  bool tryToStartBuild() {
    // Try to create the .new file
    FILE* f = fopen(moduleNew_.c_str(), "w+");
    if (!f) {
      // We lost the race to start the build
      return false;
    }
    fclose(f);
    // The .new file may look the same age as the .cpp file, but
    // the makefile is written to ignore the .new timestamp
    writeMakefile();
    writeSource();
    std::stringstream ss;
    // ss << "/usr/bin/nohup ";
    ss << "/usr/bin/make -C " << config.moduleDir_ << " -f " << moduleMak_;
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
  bool forceBuild
) :
  buildState_(kInit),
  loadState_(kInit),
  key_(hashTwoStrings(options, source)),
  isGlobal_(false),
  dlopenHandle_(nullptr),
  libName_(config.getLibName(key_)),
  newName_(config.getNewName(key_)) {
  //
  // Master constructor - try to get module built in local file
  //
  config.ensureModuleDirExists();
  if (!forceBuild && fileExists(libName_)) {
    buildState_ = kPass;
  } else {
    //
    // The file doesn't exist, let's start building it
    //
    ModuleBuilder builder(options, source, include, key_);
    builder.tryToStartBuild();
  }
}

NativeModule::NativeModule(
  bool isGlobal,
  const char* key,
  long binarySize,
  const void* binary
) :
  buildState_(isGlobal ? kPass : kInit),
  loadState_(isGlobal ? kPass : kInit),
  key_(key),
  isGlobal_(isGlobal),
  dlopenHandle_(nullptr),
  libName_(config.getLibName(key_)),
  newName_(config.getNewName(key_)) {
  //
  // Worker constructor - try to get the binary written to local file
  //
  if (isGlobal_) return;
  int rc = 0;
  config.ensureModuleDirExists();
  for (;;) {
    if (fileExists(libName_) && (fileSize(libName_) == binarySize)) {
      buildState_ = kPass;
      break;
    }
    // Race to write the new file
    int fd = open(newName_.c_str(), O_WRONLY|O_CREAT|O_EXCL|O_TRUNC, 0666);
    if (fd >= 0) {
      //
      // Now we're about to write the new file
      //
      rc = write(fd, binary, binarySize);
      assert(rc == binarySize);
      close(fd);
      ::chmod(newName_.c_str(), 0644);
      if (!fileExists(libName_)) {
        // Don't let anyone see the file until it is completely written
        rc = ::rename(newName_.c_str(), libName_.c_str());
        buildState_ = ((rc == 0) ? kPass : kFail);
        break;
      }
    } else {
      //
      // Someone else is writing to new
      //
      while (fileExistsAndIsRecent(newName_) && !fileExists(libName_)) {
        usleep(kFilePollMicrosecs);
      }
    }
  }
  if (buildState_ == kPass) tryLoad();
}

NativeModule::~NativeModule() {
  if (!isGlobal_ && dlopenHandle_) {
    dlclose(dlopenHandle_);
  }
}

bool NativeModule::tryWaitForBuild() {
  if (buildState_ == kInit) {
    //
    // The writer will rename new to lib.  If we tested exists(lib)
    // followed by exists(new) then the rename could occur between
    // the two tests. This way is safe provided that either rename is atomic,
    // or rename creates the new name before destroying the old name.
    //
    while (fileExistsAndIsRecent(newName_)) {
      usleep(kFilePollMicrosecs);
    }
    buildState_ = (fileExists(libName_) ? kPass : kFail);
    if (buildState_ == kFail) {
      std::string base(config.moduleDir_ + "/hm_" + key_);
      fprintf(stderr, "makefile:\n%s", readFileAsString(base+".mak").c_str());
      fprintf(stderr, "errors:\n%s",   readFileAsString(base+".err").c_str());
    }
  }
  return(buildState_ == kPass);
}

bool NativeModule::tryLoad() {
  if (loadState_ == kInit) {
    if (isGlobal_) {
      loadState_ = kPass;
    } else if (!tryWaitForBuild()) {
      fprintf(stderr, "libName %s tryWaitForBuild fail\n", libName_.c_str());
      loadState_ = kFail;
    } else {
      auto handle = dlopen(libName_.c_str(), RTLD_GLOBAL|RTLD_LAZY);
      if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
      }
      loadState_ = (handle ? kPass : kFail);
      if (handle) dlopenHandle_ = handle;
    }
  }
  return(loadState_ == kPass);
}

static std::string toQualifiedName(
  JNIEnv* env,
  const std::string& key,
  jstring nameJ,
  int numArgs,
  bool isGlobal
) {
  JString name(env, nameJ);
  char argTypeCodes[32];
  for (int j = 0; j < numArgs; ++j) argTypeCodes[j] = 'l';
  argTypeCodes[numArgs] = 0;  
  char buf[512];
  if (isGlobal) {
    // No name-mangling for global func names
    strcpy(buf, name);
  } else {
    auto moduleName = std::string("hm_") + key;
    sprintf(buf, "_ZN4hail%lu%s%lu%sE%s",
      moduleName.length(), moduleName.c_str(), strlen(name), (const char*)name, argTypeCodes);
  }
  return std::string(buf);
}

void NativeModule::findLongFuncL(
  JNIEnv* env,
  NativeStatus* st,
  jobject funcObj,
  jstring nameJ,
  int numArgs
) {
  void* funcAddr = nullptr;
  if (!tryLoad()) {
    NATIVE_ERROR(st, 1001, "ErrModuleNotFound");
  } else {
    auto qualName = toQualifiedName(env, key_, nameJ, numArgs, isGlobal_);
    D("isGlobal %s qualName \"%s\"\n", isGlobal_ ? "true" : "false", qualName.c_str());    
    funcAddr = ::dlsym(isGlobal_ ? RTLD_DEFAULT : dlopenHandle_, qualName.c_str());
    D("dlsym -> funcAddr %p\n", funcAddr);
    if (!funcAddr) {
      NATIVE_ERROR(st, 1003, "ErrLongFuncNotFound dlsym(\"%s\")", qualName.c_str());
    }
  }
  auto ptr = MAKE_NATIVE(NativeFuncObj<long>, shared_from_this(), funcAddr);
  initNativePtr(env, funcObj, &ptr);
}

void NativeModule::findPtrFuncL(
  JNIEnv* env,
  NativeStatus* st,
  jobject funcObj,
  jstring nameJ,
  int numArgs
) {
  void* funcAddr = nullptr;
  if (!tryLoad()) {
    NATIVE_ERROR(st, 1001, "ErrModuleNotFound");
  } else {
    auto qualName = toQualifiedName(env, key_, nameJ, numArgs, isGlobal_);
    funcAddr = ::dlsym(isGlobal_ ? RTLD_DEFAULT : dlopenHandle_, qualName.c_str());
    if (!funcAddr) {
      NATIVE_ERROR(st, 1003, "ErrPtrFuncNotFound dlsym(\"%s\")", qualName.c_str());
    }
  }
  auto ptr = MAKE_NATIVE(NativeFuncObj<NativeObjPtr>, shared_from_this(), funcAddr);
  initNativePtr(env, funcObj, &ptr);
}

//
// Functions implementing NativeModule native methods
//
static NativeModule* toNativeModule(JNIEnv* env, jobject obj) {
  // It should be a dynamic_cast, but I'm trying to eliminate
  // the use of RTTI which is problematic in dynamic libraries
  return reinterpret_cast<NativeModule*>(getFromNativePtr(env, obj));
}

NATIVEMETHOD(void, NativeModule, nativeCtorMaster)(
  JNIEnv* env,
  jobject thisObj,
  jstring optionsJ,
  jstring sourceJ,
  jstring includeJ,
  jboolean forceBuildJ
) {
  JString options(env, optionsJ);
  JString source(env, sourceJ);
  JString include(env, includeJ);
  bool forceBuild = (forceBuildJ != JNI_FALSE);
  auto ptr = MAKE_NATIVE(NativeModule, options, source, include, forceBuild);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativeModule, nativeCtorWorker)(
  JNIEnv* env,
  jobject thisObj,
  jboolean isGlobalJ,
  jstring keyJ,
  jbyteArray binaryJ
) {
  bool isGlobal = (isGlobalJ != JNI_FALSE);
  JString key(env, keyJ);
  long binarySize = env->GetArrayLength(binaryJ);
  auto binary = env->GetByteArrayElements(binaryJ, 0);
  auto ptr = MAKE_NATIVE(NativeModule, isGlobal, key, binarySize, binary);
  env->ReleaseByteArrayElements(binaryJ, binary, JNI_ABORT);
  initNativePtr(env, thisObj, &ptr);
}

NATIVEMETHOD(void, NativeModule, nativeFindOrBuild)(
  JNIEnv* env,
  jobject thisObj,
  long stAddr
) {
  auto mod = toNativeModule(env, thisObj);
  auto st = reinterpret_cast<NativeStatus*>(stAddr);
  st->clear();
  if (!mod->tryWaitForBuild()) {
    NATIVE_ERROR(st, 1004, "ErrModuleBuildFailed");
  }
}

NATIVEMETHOD(jstring, NativeModule, getKey)(
  JNIEnv* env,
  jobject thisObj
) {
  auto mod = toNativeModule(env, thisObj);
  return env->NewStringUTF(mod->key_.c_str());
}

NATIVEMETHOD(jbyteArray, NativeModule, getBinary)(
  JNIEnv* env,
  jobject thisObj
) {
  auto mod = toNativeModule(env, thisObj);
  int fd = open(config.getLibName(mod->key_).c_str(), O_RDONLY, 0666);
  if (fd < 0) {
    perror("open");
    return env->NewByteArray(0);
  }
  struct stat st;
  int rc = fstat(fd, &st);
  assert(rc == 0);
  size_t fileSize = st.st_size;
  jbyteArray result = env->NewByteArray(fileSize);
  jbyte* resultBuf = env->GetByteArrayElements(result, 0);
  rc = read(fd, resultBuf, fileSize);
  assert(rc == (int)fileSize);
  close(fd);
  env->ReleaseByteArrayElements(result, resultBuf, 0);
  return result;
}

#define DECLARE_FIND(LongOrPtr, numArgs) \
NATIVEMETHOD(void, NativeModule, nativeFind##LongOrPtr##FuncL##numArgs)( \
  JNIEnv* env, \
  jobject thisObj, \
  long stAddr, \
  jobject funcObj, \
  jstring nameJ \
) { \
  auto mod = toNativeModule(env, thisObj); \
  auto st = reinterpret_cast<NativeStatus*>(stAddr); \
  st->clear(); \
  mod->find##LongOrPtr##FuncL(env, st, funcObj, nameJ, numArgs); \
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

NAMESPACE_END(hail)
