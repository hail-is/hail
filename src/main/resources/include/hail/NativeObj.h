#ifndef HAIL_NATIVEOBJ_H
#define HAIL_NATIVEOBJ_H 1

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

// Declare a function to be exported from a DLL with "C" linkage

#define NATIVEMETHOD(cppReturnType, scalaClass, scalaMethod) \
  extern "C" __attribute__((visibility("default"))) \
    cppReturnType Java_is_hail_nativecode_##scalaClass##_##scalaMethod

namespace hail {

// The lifetime of off-heap objects may be controlled by any combination
// of C++ std::shared_ptr<T> and Jvm-side NativePtr objects.  But classes
// used in this way must inherit from NativeObj.

class NativeObj :
  public std::enable_shared_from_this<NativeObj> {
  public:
    // Objects managed with a Jvm-side NativePtr need to be destroyed
    // without knowing their precise type, so they must have a virtual
    // destructor.
    virtual ~NativeObj() { }

    // Subclasses should override getClassName, which may be helpful
    // for debugging.
    virtual const char* get_class_name() { return "NativeObj"; }
    
    // If a subclass wants to make some fields visible to Jvm Unsafe
    // access, it can publish their offsets.  This is probably only
    // useful for integer types, so we pass in one parameter to
    // identify the fieldSize as 1, 2, 4, or 8 bytes.
    virtual int64_t get_field_offset(int fieldSize, const char* fieldName) { return(-1); }
};

// On the Jvm side, we don't have distinct classes/types for NativePtr
// to different off-heap objects - we treat them all as NativeObjPtr.
using NativeObjPtr = std::shared_ptr<NativeObj>;

// There are at least three different implementations of std::string,
// one in libc++, one in libstdc++ for abi-version <= 8, and another
// for abi-version >= 9.  To avoid possible conflicts between prebuilt
// libraries and dynamically-generated code, we use our own minimal
// hail::hstring in public interfaces.

class hstring {
 private:
  uint32_t length_;
  uint32_t buf_size_;
  char* buf_;

 private:
  void init(const char* s, size_t len) {
    length_ = len;
    buf_size_ = ((len+1+0xf) & ~0xf);
    buf_ = (char*)malloc(buf_size_);
    memcpy(buf_, s, len);
    buf_[len] = 0;
  }
  
  hstring& reinit(const char* s, size_t len) {
    length_ = len;
    if (len+1 > buf_size_) {
      if (buf_) free(buf_);
      buf_size_ = ((length_+1+0xf) & ~0xf);
      buf_ = (char*)malloc(buf_size_);
    }
    memcpy(buf_, s, len);
    buf_[len] = 0;
    return *this;
  }

 public:
  hstring() { init(nullptr, 0); }
  
  hstring(const hstring& b) { init(b.buf_, b.length_); }
  
  hstring(const char* s) { init(s, strlen(s)); }
  
  hstring(const std::string& s) { init(s.data(), s.length()); }
  
  ~hstring() { if (buf_) free(buf_); }

  hstring& operator=(const hstring& b) { return reinit(b.buf_, b.length_); }
    
  hstring& operator=(const char* s) { return reinit(s, strlen(s)); }
  
  hstring& operator=(const std::string& s) { return reinit(s.data(), s.length()); }
  
  size_t length() { return length_; }
  
  const char* c_str() const { return buf_; }
  
  // This converts to whatever std::string is being used in the caller
  operator std::string() const { return std::string(buf_, length_); }
};

} // end hail

#endif
