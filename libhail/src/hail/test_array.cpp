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
    auto vint64 = cast<VInt64>(tc.get_vtype(tc.tint64));
    auto vfloat64 = cast<VFloat64>(tc.get_vtype(tc.tfloat64));
    auto vstr = cast<VStr>(tc.get_vtype(tc.tstr));
    auto vbool = cast<VBool>(tc.get_vtype(tc.tbool));

    // TEST_CASE(test_array_length) {
    //     auto region = std::make_shared<ArenaAllocator>(heap);
    //     auto varray = cast<VArray>(tc.get_vtype(tc.tarray(tc.tfloat64)));

    //     int array_length = 8;
    //     auto my_array = Value::make_array(varray, region, array_length);
    //     assert(array_length == my_array.get_size());

    //     for (int i = 0; i < array_length; ++i) {
    //         Value element(vfloat64, 5.2 + i);
    //         my_array.set_element(i, element);
    //     }
    //     print(my_array);

    //     IRContext xc(heap);
    //     Module *m = xc.make_module();
    //     JIT jit;

    //     std::vector<const Type *> param_types;
    //     std::vector<const VType *> param_vtypes;
    //     std::vector<IR *> array_elements;

    //     auto return_type = tc.tint64;
    //     const VType *return_vtype = tc.get_vtype(return_type);

    //     Function *length_check = xc.make_function(m, "main", param_types, return_type);
    //     auto body = length_check->get_body();

    //     for (int i = 0; i < array_length; ++i) {
    //         Value element(vfloat64, 5.2 + i);
    //         array_elements.push_back(body->make_literal(element));
    //     }
    //     print("Added elements to array");

    //     body->set_child(0, body->make_array_len(body->make_make_array(array_elements)));

    //     for (auto t : param_types)
    //         param_vtypes.push_back(tc.get_vtype(t));

    //     auto compiled = jit.compile(heap, tc, m, param_vtypes, return_vtype);
    //     auto length_check_return_value = compiled.invoke(region, {});
    //     assert(length_check_return_value.as_int64() == array_length);
    // }

    TEST_CASE(test_array_ref) {
        IRContext xc(heap);
        Module *m = xc.make_module();
        JIT jit;
        std::vector<const Type *> param_types;
        std::vector<const VType *> param_vtypes;
        std::vector<IR *> array_elements;
        auto region = std::make_shared<ArenaAllocator>(heap);

        auto return_type = tc.tfloat64;
        const VType *return_vtype = tc.get_vtype(return_type);

        Function *ref_check = xc.make_function(m, "main", param_types, return_type);
        auto body = ref_check->get_body();

        for (int i = 0; i < 5; ++i) {
            Value element(vfloat64, 5.2 + i);
            array_elements.push_back(body->make_literal(element));
        }

        auto idx_value = Value(vint64, 1);
        body->set_child(0, body->make_array_ref(body->make_make_array(array_elements), body->make_literal(idx_value)));

        for (auto t : param_types)
            param_vtypes.push_back(tc.get_vtype(t));

        auto compiled = jit.compile(heap, tc, m, param_vtypes, return_vtype);
        auto ans = compiled.invoke(region, {});
        //assert(ans.as_float64() == 6.2);
    }
}