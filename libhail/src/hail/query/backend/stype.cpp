#include <hail/query/backend/stype.hpp>

namespace hail {

SType::~SType() {}

SValue *
SType::from_llvm_values(std::vector<llvm::Value *> &llvm_values, size_t i) const {
  switch (tag) {
  case SType::Tag::BOOL:
    return new SBoolValue(this, llvm_values[i]);
  case SType::Tag::INT32:
    return new SInt32Value(this, llvm_values[i]);
  case SType::Tag::INT64:
    return new SInt64Value(this, llvm_values[i]);
  case SType::Tag::FLOAT32:
    return new SFloat32Value(this, llvm_values[i]);
  case SType::Tag::FLOAT64:
    return new SFloat64Value(this, llvm_values[i]);
  }
}

SBool::SBool(const Type *type)
  : SType(self_tag, type) {}

std::vector<PrimitiveType>
SBool::constituent_types() const {
  return {PrimitiveType::INT8};
}

SInt32::SInt32(const Type *type)
  : SType(self_tag, type) {}

std::vector<PrimitiveType>
SInt32::constituent_types() const {
  return {PrimitiveType::INT32};
}

SInt64::SInt64(const Type *type)
  : SType(self_tag, type) {}

std::vector<PrimitiveType>
SInt64::constituent_types() const {
  return {PrimitiveType::INT64};
}

SFloat32::SFloat32(const Type *type)
  : SType(self_tag, type) {}

std::vector<PrimitiveType>
SFloat32::constituent_types() const {
  return {PrimitiveType::FLOAT32};
}

SFloat64::SFloat64(const Type *type)
  : SType(self_tag, type) {}

std::vector<PrimitiveType>
SFloat64::constituent_types() const {
  return {PrimitiveType::FLOAT64};
}

EmitValue
EmitType::from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) {
  llvm::Value *missing = llvm_values[i++];
  SValue *svalue = stype->from_llvm_values(llvm_values, i);
  return EmitValue(missing, svalue);
}

}
