#include "hail/format.hpp"
#include "hail/test.hpp"

namespace hail {

Test::Test(std::string name, TestFunction *function)
  : name(std::move(name)), function(function), next(Tests::head) {
  Tests::head = this;
}

int
Test::run() const {
  format(errs, "RUN ", name, "\n");
  function();
  format(errs, "RUN ", name, " OK\n");
  return 0;
}

Test *Tests::head = nullptr;

int
Tests::run_tests() {
  for (const Test *t = head; t; t = t->next) {
    t->run();
  }
  return 0;
}

}

int
main() {
  hail::Tests::run_tests();
}
