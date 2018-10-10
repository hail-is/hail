#ifndef HAIL_LOGGING_H
#define HAIL_LOGGING_H 1

#include <string>

// Upcalls from C++ to Scala is.hail.utils.{info,warn,error}
//
// In namespace hail rather than hail::Logging, so that we can use 
// {info,warn,error} in generated code without qualifiying as "Logging.info"

namespace hail {
  
void set_test_msg(const char* msg);

void info(const char* msg);

void warn(const char* msg);

void error(const char* msg);

// inline functions convert from either old-ABI or new-ABI std::string

static inline void set_test_msg(const std::string& msg) { set_test_msg(msg.c_str()); }

static inline void info(const std::string& msg) { info(msg.c_str()); }

static inline void warn(const std::string& msg) { warn(msg.c_str()); }

static inline void error(const std::string& msg) { error(msg.c_str()); }

}

#endif
