#ifndef HAIL_UPCALLS_H
#define HAIL_UPCALLS_H 1

#include <jni.h>
#include <cstdint>
#include <string>

namespace hail {

// UpcallConfig holds jmethodID's for various class methods
class UpcallConfig {
 public:
  JavaVM* java_vm_;
  jobject upcalls_;
  // Upcalls methods
  jmethodID Upcalls_setTestMsg_;
  jmethodID Upcalls_info_;
  jmethodID Upcalls_warn_;
  jmethodID Upcalls_error_;
  // InputStream methods
  jmethodID InputStream_close_; // close()
  jmethodID InputStream_read_;  // read(buf: Array[Byte], off: int, len: Int): Int
  jmethodID InputStream_skip_;  // skip(len: Long): Long
  // OutputStream methods
  jmethodID OutputStream_close_;
  jmethodID OutputStream_flush_;
  jmethodID OutputStream_write_;
  // RVIterator methods
  jmethodID RVIterator_hasNext_;
  jmethodID RVIterator_next_;

  UpcallConfig();
};

class UpcallEnv {
 private:
  const UpcallConfig* config_; // once-per-session jobject/classID/methodID's
  JNIEnv* env_;
  bool did_attach_;
  
 public:
  // Constructor ensures thread is attached to JavaVM, and gets a JNIEnv 
  UpcallEnv();

  // Destructor restores the previous state
  ~UpcallEnv();
  
  const UpcallConfig* config() { return config_; }
  
  JNIEnv* env() const { return env_; }
  
  // Test with same interface as logging calls 
  void set_test_msg(const std::string& msg);
  
  // Logging (through is.hail.utils)
  void info(const std::string& msg);
  void warn(const std::string& msg);
  void error(const std::string& msg);

};

} // end hail

#endif
