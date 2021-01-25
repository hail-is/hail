#include <hail/allocators.hpp>
#include <hail/format.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>

#include <cstdio>

using namespace hail;

class A {
public:
  A() { println("ctor"); }
  ~A() { println("dtor"); }
};

int
main() {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);
  auto a = arena.make<A>();
  auto ab = tc.tarray(tc.tbool);

  println("this: ", 5, " is a number and this: ", ab);
}
