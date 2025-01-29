#include "hail/Logging.h"
#include "hail/Upcalls.h"
#include <cstdio>
#include <string>

namespace hail {

void set_test_msg(const std::string& msg) {
  fprintf(stderr, "DEBUG: Logging set_test_msg ...\n");
  UpcallEnv e;
  e.set_test_msg(msg);
}

void info(const std::string& msg) {
  UpcallEnv e;
  e.info(msg);
}

void warn(const std::string& msg) {
  UpcallEnv e;
  e.warn(msg);
}

void error(const std::string& msg) {
  UpcallEnv e;
  e.error(msg);
}

}
