#ifndef HAIL_TEST_HPP_INCLUDED
#define HAIL_TEST_HPP_INCLUDED 1

#include <string>
#include <vector>

#include "hail/format.hpp"

namespace hail {

using TestFunction = void();

class Test {
  friend class Tests;

  const std::string name;
  TestFunction *const function;
  Test *next;

public:
  Test(std::string name, TestFunction *function);

  int run() const;
};

class Tests {
  friend class Test;

  static Test *head;

public:
  static int run_tests();
};

#define TEST_CASE(test_name)				\
static void test_name();				\
static Test test_name##_obj(#test_name, test_name);	\
void test_name()

template<typename L, typename R> void
check_eq_impl(const char *lstr, const char *rstr, const L &l, const R &r) {
  if (l != r) {
    format(errs, __FILE__, ":", __LINE__, ": assert failed:\n");
    format(errs, "  CHECK_EQ(", lstr, ", ", rstr, ")\n");
    format(errs, "with values:\n");
    // FIXME what if l, r don't support format?
    format(errs, "  CHECK_EQ(", l, ", ", r, ")\n");
    // FIXME throw so the rest of the tests run
    abort();
  }
}

#define CHECK_EQ(l, r)	check_eq_impl(#l, #r, l, r)

}

#endif
