#ifndef HAIL_HSTRING_H
#define HAIL_HSTRING_H 1

#include <cstdint>
#include <string>
#include <vector>

namespace hail {

// Classes hail::hstring and hail::hstringstream provide a subset of
// the functionality of std::string and std::stringstream, avoiding
// the ABI incompatibility between a libhail.so built with g++-5.x or later,
// and a libstdc++.so and dynamically-generated code built with
// g++-4.x.
//
// This is needed in particular to support the default google dataproc
// environment, currently based on debian8 (Jessie) with g++-4.9.x
//
// It's fine for dynamically-generated code to use std::string.  The
// trouble arises when we have a new-ABI-std::string used in
// prebuilt libhail.so, trying to link against an old-ABI libstdc++;
// or a dynamically-generated module using old-ABI-std::string trying
// to link against a libhail.so built with new-ABI-std::string.

class hstring {
 private:
  static const size_t kMinCapacity = 15;
 private:
  size_t capacity_;
  size_t length_;
  char* buf_;

 private:
  void init(const char* s, size_t n);
  
  void assign(const char* s, size_t n);
  
 public:
  hstring() : capacity_(0), length_(0), buf_(nullptr) { }

  hstring(const char* s, size_t n);

  hstring(const char* s);
  
  hstring(const hstring& b);
  
  hstring(hstring&& b);

  ~hstring();
  
  hstring& operator=(const char* s);
  
  hstring& operator=(const hstring& b);
  
  hstring& operator=(hstring&& b);
  
  // Comparison operators (needed for use in std::map)
  
  bool operator==(const hstring& b) const;
  
  bool operator!=(const hstring& b) const;
  
  bool operator<(const hstring& b) const;

  // Inline methods for dealing with std::string will work whether the
  // caller is using old-ABI or new-ABI std::string
    
  inline hstring(const std::string& s) : capacity_(0), length_(0), buf_(nullptr) {
    init(s.c_str(), s.length());
  }
  
  inline hstring& operator=(const std::string& b) {
    return (*this = b.c_str());
  }
  
  inline operator std::string() const {
    return std::string(buf_, length_);
  }
  
  size_t size() const { return length_; }
  
  size_t length() const { return length_; }
  
  size_t capacity() const { return capacity_; }
  
  void reserve(size_t min_cap);

  void clear() {
    if (length_ > 0) { length_ = 0; buf_[0] = 0; }
  }
  
  bool empty() const { return (length_ == 0); }
  
  const char* data() const { return buf_; }
  
  char* data() { return buf_; }
  
  const char* c_str() const {
    return ((length_ == 0) ? "" : buf_);
  }
  
  hstring operator+(const hstring& b) const;
  
  void append(const char* s, size_t n);
};

class hstringstream {
 private:
  hstring str_;
  
 public:
  const hstring& str() const { return str_; }
  
  hstringstream& operator<<(const hstring& b);
  
  hstringstream& operator<<(const char* s);
  
  hstringstream& operator<<(uint64_t n);
  
  hstringstream& operator<<(int64_t n);
  
  hstringstream& operator<<(char c);
};

} // end hail

#endif
