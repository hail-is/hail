#include <cstdio>
#include <cstring>

#include <memory>

#include <hail/allocators.hpp>
#include <hail/format.hpp>
#include <hail/query/backend/jit.hpp>
#include <hail/query/ir.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>
#include <hail/value.hpp>

using namespace hail;

int
main() {
  HeapAllocator heap;
  ArenaAllocator arena(heap);
  TypeContext tc(heap);

  auto vint32 = cast<VInt32>(tc.get_vtype(tc.tint32));
  auto vstr = cast<VStr>(tc.get_vtype(tc.tstr));
  auto vbool = cast<VBool>(tc.get_vtype(tc.tbool));

  {
    auto t = tc.ttuple({tc.tint32, tc.tstr});

    print("this: ", 5, " is a number and this is a type: ", t);

    auto region = std::make_shared<ArenaAllocator>(heap);

    auto vt = cast<VTuple>(tc.get_vtype(t));

    Value i(vint32, 5);
    auto s = Value::make_str(vstr, region, 5);
    assert(s.get_size() == 5);
    memcpy(s.get_data(), "fooba", 5);
    auto tv = Value::make_tuple(vt, region);
    tv.set_element_missing(0, false);
    tv.set_element(0, i);
    tv.set_element_missing(1, false);
    tv.set_element(1, s);

    print("tv = ", tv);
  }

  {
    auto region = std::make_shared<ArenaAllocator>(heap);
    IRContext xc(heap);

    Module *m = xc.make_module();

    std::vector<const Type *> param_types;
    const Type *return_type = tc.tbool;

    Function *f = xc.make_function(m, "main", param_types, return_type);
    auto body = f->get_body();
    body->set_child(0, body->make_na(tc.tbool));

    m->pretty_self(outs);

    JIT jit;


    std::vector<const VType *> param_vtypes;
    for (auto t : param_types)
      param_vtypes.push_back(tc.get_vtype(t));
    const VType *return_vtype = tc.get_vtype(return_type);

    auto compiled = jit.compile(tc, m, param_vtypes, return_vtype);

    auto return_value = compiled.invoke(region, {});
    print("return_value: ", return_value);
  }

  {
    auto region = std::make_shared<ArenaAllocator>(heap);
    auto varray = cast<VArray>(tc.get_vtype(tc.tarray(tc.tfloat64)));
  }

  return 0;
}
