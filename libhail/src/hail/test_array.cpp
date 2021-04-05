#include <hail/allocators.hpp>
#include <hail/test.hpp>
#include <hail/type.hpp>
#include <hail/value.hpp>
#include <hail/vtype.hpp>
#include <hail/query/ir.hpp>
#include <hail/query/backend/jit.hpp>

namespace hail {
    HeapAllocator heap;
    ArenaAllocator arena(heap);
    TypeContext tc(heap);

    auto vint32 = cast<VInt32>(tc.get_vtype(tc.tint32));
    auto vfloat64 = cast<VFloat64>(tc.get_vtype(tc.tfloat64));
    auto vstr = cast<VStr>(tc.get_vtype(tc.tstr));
    auto vbool = cast<VBool>(tc.get_vtype(tc.tbool));

    TEST_CASE(test_array_compile) {
        print("Array compile testing");
        // auto region = std::make_shared<ArenaAllocator>(heap);
        // IRContext xc(heap);

        // Module *m = xc.make_module();

        // std::vector<const VType *> param_vtypes;
        // auto return_type = tc.tint64;
        // const VType *return_vtype = tc.get_vtype(return_type);

        // JIT jit;

        // auto compiled = jit.compile(heap, tc, m, param_vtypes, return_vtype);

        //auto return_value = compiled.invoke(region, {});
        //print("return_value: ", return_value);
    }
}