#ifndef HAIL_VTYPE_HPP_INCLUDED
#define HAIL_VTYPE_HPP_INCLUDED 1

#include <hail/type.hpp>

namespace hail {

class VType {
public:
  using BaseType = VType;
  enum class Tag {
    VOID,
    BOOL,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STR,
    ARRAY,
    STREAM,
    TUPLE
  };
  const Tag tag;
  size_t alignment;
  size_t byte_size;
  VType(Tag tag, size_t alignment, size_t byte_size) : tag(tag), alignment(alignment), byte_size(byte_size) {}
  virtual ~VType();
};

class VBool : public VType {
public:
  static const Tag self_tag = VType::Tag::BOOL;
  VBool(TypeContextToken) : VType(self_tag, 1, 1) {}
};

class VInt32 : public VType {
public:
  static const Tag self_tag = VType::Tag::BOOL;
  VInt32(TypeContextToken) : VType(self_tag, 4, 4) {}
};

class VInt64 : public VType {
public:
  static const Tag self_tag = VType::Tag::BOOL;
  VInt64(TypeContextToken) : VType(self_tag, 8, 8) {}
};

class VFloat32 : public VType {
public:
  static const Tag self_tag = VType::Tag::BOOL;
  VFloat32(TypeContextToken) : VType(self_tag, 4, 4) {}
};

class VFloat64 : public VType {
public:
  static const Tag self_tag = VType::Tag::BOOL;
  VFloat64(TypeContextToken) : VType(self_tag, 8, 8) {}
};

class VStr : public VType {
public:
  static const Tag self_tag = VType::Tag::STR;
  VStr(TypeContextToken) : VType(self_tag, 8, 8) {}
};

class VArray : public VType {
public:
  static const Tag self_tag = VType::Tag::ARRAY;
  const VType *const element_vtype;
  size_t elements_alignment;
  size_t element_stride;
  VArray(TypeContextToken, const VType *element_vtype)
    : VType(self_tag, 8, 8),
      element_vtype(element_vtype),
      elements_alignment(make_aligned(element_vtype->byte_size, element_vtype->alignment)),
      element_stride(make_aligned(element_vtype->byte_size, element_vtype->alignment)) {}
};

class VTuple : public VType {
  friend class TypeContext;
public:
  static const Tag self_tag = VType::Tag::TUPLE;
  const std::vector<const VType *> element_vtypes;
  const std::vector<size_t> element_offsets;
  VTuple(TypeContextToken,
	 std::vector<const VType *> element_vtypes,
	 std::vector<size_t> element_offsets,
	 size_t alignment,
	 size_t byte_size)
    : VType(self_tag, alignment, byte_size),
      element_vtypes(element_vtypes),
      element_offsets(element_offsets) {}
};

}

#endif
