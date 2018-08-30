#ifndef HAIL_NATIVESTATUS_H
#define HAIL_NATIVESTATUS_H 1

#include "hail/NativeObj.h"
#include <cstdarg>
#include <cstdio>
#include <string>

namespace hail {

class NativeStatus : public NativeObj {
public:
  int errno_;
  std::string msg_;
  std::string location_;
  
public:
  inline NativeStatus() : errno_(0) { }
  
  virtual ~NativeStatus() { }
  
  inline void clear() {
    // When errno_ == 0, the values of msg_ and location_ are ignored
    errno_ = 0;
  }
  
  void set(const char* file, int line, int code, const char* msg, ...) {
    char buf[8*1024];
    sprintf(buf, "%s,%d", file, line);
    location_ = buf;
    errno_ = code;
    va_list argp;
    va_start(argp, msg);
    vsprintf(buf, msg, argp);
    va_end(argp);
    msg_ = buf;
  }
};

using NativeStatusPtr = std::shared_ptr<NativeStatus>;

#define NATIVE_ERROR(_p, _code, _msg, ...) \
   { (_p)->set(__FILE__, __LINE__, _code, _msg, ##__VA_ARGS__); }

} // end hail

#endif
