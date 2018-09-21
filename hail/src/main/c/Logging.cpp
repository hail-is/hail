#include "hail/Logging.h"
#include "hail/Upcalls.h"
#include <cstdio>
#include <string>

namespace hail {

void set_test_msg(const char* msg) {
  UpcallEnv e;
  e.set_test_msg(msg);
}

void info(const char* msg) {
  UpcallEnv e;
  e.info(msg);
}

void warn(const char* msg) {
  UpcallEnv e;
  e.warn(msg);
}

void error(const char* msg) {
  UpcallEnv e;
  e.error(msg);
}

}
