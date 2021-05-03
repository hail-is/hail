#include "hail/allocators.hpp"
#include "hail/test.hpp"
#include "hail/type.hpp"
#include "hail/value.hpp"
#include "hail/vtype.hpp"

namespace hail {

TEST_CASE(test_bool_value) {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);
  auto vbool = cast<VBool>(tc.get_vtype(tc.tbool));

  Value tv(vbool, true);
  assert(!tv.get_missing());
  assert(tv.as_bool());

  Value fv(vbool, false);
  assert(!fv.get_missing());
  assert(!fv.as_bool());

  Value mv(vbool);
  assert(mv.get_missing());
}

TEST_CASE(test_int32_value) {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);
  auto vint32 = cast<VInt32>(tc.get_vtype(tc.tint32));

  Value iv(vint32, 5);
  assert(!iv.get_missing());
  assert(iv.as_int32() == 5);

  Value mv(vint32);
  assert(mv.get_missing());
}

TEST_CASE(test_int64_value) {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);
  auto vint64 = cast<VInt64>(tc.get_vtype(tc.tint64));

  Value iv(vint64, 5);
  assert(!iv.get_missing());
  assert(iv.as_int64() == 5);

  Value mv(vint64);
  assert(mv.get_missing());
}

}
