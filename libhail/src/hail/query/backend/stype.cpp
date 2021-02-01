#include <hail/query/backend/stype.hpp>

namespace hail {

SType::~SType() {}

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

}
