#ifndef HAIL_LOGGING_H
#define HAIL_LOGGING_H 1
#include <string>

// Upcalls from C++ to Scala is.hail.utils.{info,warn,error}
//
// In namespace hail rather than hail::Logging, so that we can use 
// {info,warn,error} in generated code without qualifiying as "Logging.info"

namespace hail {
  
void set_test_msg(const std::string& msg);

void info(const std::string& msg);

void warn(const std::string& msg);

void error(const std::string& msg);

}

#endif
