#include <hail/query/backend/svalue.hpp>

namespace hail {

SValue::~SValue() {}

SBoolValue::SBoolValue(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}


SInt32Value::SInt32Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

SInt64Value::SInt64Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

SFloat32Value::SFloat32Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

SFloat64Value::SFloat64Value(const SType *stype, llvm::Value *value)
  : SValue(self_tag, stype), value(value) {}

EmitValue::EmitValue(llvm::Value *missing, const SValue *svalue) {
  abort();
}

EmitValue::EmitValue(llvm::BasicBlock *present_block, llvm::BasicBlock *missing_block, const SValue *svalue) {
  abort();
}

EmitControlValue
EmitValue::as_control() const {
  abort();
}

EmitDataValue
EmitValue::as_data() const {
  abort();
}


}
