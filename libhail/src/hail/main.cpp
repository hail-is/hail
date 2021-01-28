#include <cstdio>
#include <cstring>

#include <memory>

#include <hail/allocators.hpp>
#include <hail/format.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>
#include <hail/value.hpp>

using namespace hail;

int
main() {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);
  auto t = tc.ttuple({tc.tint32, tc.tstr});

  print("this: ", 5, " is a number and this is a type: ", t);

  auto region = std::make_shared<ArenaAllocator>(heap);

  auto p = cast<PTuple>(tc.get_canonical_ptype(t));
  auto pint32 = cast<PInt32>(p->element_ptypes[0]);
  auto pstr = cast<PStr>(p->element_ptypes[1]);

  Value i(pint32, 5);
  auto s = Value::make_str(pstr, region, 5);
  assert(s.get_size() == 5);
  memcpy(s.get_data(), "fooba", 5);
  auto v = Value::make_tuple(p, region);
  v.set_element_present(0, true);
  v.set_element(0, i);
  v.set_element_present(1, true);
  v.set_element(1, s);

  print("v = ", v);

  return 0;
}
