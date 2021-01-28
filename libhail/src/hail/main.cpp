#include <hail/allocators.hpp>
#include <hail/format.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>

#include <cstdio>

using namespace hail;

class A {
public:
  A() { print("ctor"); }
  ~A() { print("dtor"); }
};

int
main() {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);
  auto a = arena.make<A>();
  auto ab = tc.ttuple({tc.tbool, tc.tarray(tc.tint32)});

  print("this: ", 5, " is a number and this: ", ab);
}
