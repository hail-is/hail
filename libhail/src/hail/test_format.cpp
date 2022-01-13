#include <cassert>

#include "hail/allocators.hpp"
#include "hail/format.hpp"
#include "hail/test.hpp"
#include "hail/type.hpp"
#include "hail/value.hpp"
#include "hail/vtype.hpp"

namespace hail {

TEST_CASE(test_render) {
  assert(render(5) == "5");

  std::string s = "foo";
  assert(render(5, s, 2.5) == "5foo2.5");
}

TEST_CASE(test_format_address) {
  CHECK_EQ(render(FormatAddress(nullptr)), "0000000000000000");
}

TEST_CASE(test_indent) {
  assert(render(Indent(5), "and then") == "     and then");
}

}
