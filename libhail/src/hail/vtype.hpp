#ifndef HAIL_VTYPE_HPP_INCLUDED
#define HAIL_VTYPE_HPP_INCLUDED 1

#include <hail/type.hpp>

namespace hail {

class TBool;
class TInt32;
class TInt64;
class TFloat32;
class TFloat64;
class TStr;
class TArray;
class TTuple;

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
  const Type *type;
  size_t alignment;
  size_t byte_size;
  VType(Tag tag, const Type *type, size_t alignment, size_t byte_size)
    : tag(tag),
      type(type),
      alignment(alignment),
      byte_size(byte_size) {}
  virtual ~VType();
};

class VBool : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::BOOL; }
  VBool(TypeContextToken, const TBool *type) : VType(VType::Tag::BOOL, type, 1, 1) {}
};

class VInt32 : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::INT32; }
  VInt32(TypeContextToken, const TInt32 *type) : VType(VType::Tag::INT32, type, 4, 4) {}
};

class VInt64 : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::INT64; }
  VInt64(TypeContextToken, const TInt64 *type) : VType(VType::Tag::INT64, type, 8, 8) {}
};

class VFloat32 : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::FLOAT32; }
  VFloat32(TypeContextToken, const TFloat32 *type) : VType(VType::Tag::FLOAT32, type, 4, 4) {}
};

class VFloat64 : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::FLOAT64; }
  VFloat64(TypeContextToken, const TFloat64 *type) : VType(VType::Tag::FLOAT64, type, 8, 8) {}
};

class VStr : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::STR; }
  VStr(TypeContextToken, const TStr *type) : VType(VType::Tag::STR, type, 8, 8) {}
};

class VArray : public VType {
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::ARRAY; }
  const VType *const element_vtype;
  size_t elements_alignment;
  size_t element_stride;
  VArray(TypeContextToken, const TArray *type, const VType *element_vtype)
    : VType(VType::Tag::ARRAY, type, 8, 8),
      element_vtype(element_vtype),
      elements_alignment(make_aligned(element_vtype->byte_size, element_vtype->alignment)),
      element_stride(make_aligned(element_vtype->byte_size, element_vtype->alignment)) {}
};

class VTuple : public VType {
  friend class TypeContext;
public:
  static bool is_instance_tag(Tag tag) { return tag == VType::Tag::TUPLE; }
  const std::vector<const VType *> element_vtypes;
  const std::vector<size_t> element_offsets;
  VTuple(TypeContextToken,
	 const TTuple *type,
	 std::vector<const VType *> element_vtypes,
	 std::vector<size_t> element_offsets,
	 size_t alignment,
	 size_t byte_size)
    : VType(VType::Tag::TUPLE, type, alignment, byte_size),
      element_vtypes(element_vtypes),
      element_offsets(element_offsets) {}
};

}

#endif
