#ifndef HAIL_TEST_HPP_INCLUDED
#define HAIL_TEST_HPP_INCLUDED 1

#include <string>
#include <vector>

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

}

#endif
