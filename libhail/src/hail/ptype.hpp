#ifndef HAIL_PTYPE_HPP_INCLUDED
#define HAIL_PTYPE_HPP_INCLUDED 1

#include <hail/type.hpp>

namespace hail {

class PType {
public:
  using BaseType = PType;
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
  PType(Tag tag, size_t alignment, size_t byte_size) : tag(tag), alignment(alignment), byte_size(byte_size) {}
  virtual ~PType();
};

class PBool : public PType {
public:
  static const Tag self_tag = PType::Tag::BOOL;
  PBool(TypeContextToken) : PType(self_tag, 1, 1) {}
};

class PInt32 : public PType {
public:
  static const Tag self_tag = PType::Tag::BOOL;
  PInt32(TypeContextToken) : PType(self_tag, 4, 4) {}
};

class PInt64 : public PType {
public:
  static const Tag self_tag = PType::Tag::BOOL;
  PInt64(TypeContextToken) : PType(self_tag, 8, 8) {}
};

class PFloat32 : public PType {
public:
  static const Tag self_tag = PType::Tag::BOOL;
  PFloat32(TypeContextToken) : PType(self_tag, 4, 4) {}
};

class PFloat64 : public PType {
public:
  static const Tag self_tag = PType::Tag::BOOL;
  PFloat64(TypeContextToken) : PType(self_tag, 8, 8) {}
};

class PStr : public PType {
public:
  static const Tag self_tag = PType::Tag::STR;
  PStr(TypeContextToken) : PType(self_tag, 8, 8) {}
};

class PArray : public PType {
public:
  static const Tag self_tag = PType::Tag::ARRAY;
  const PType *const element_ptype;
  size_t elements_alignment;
  size_t element_stride;
  PArray(TypeContextToken, const PType *element_ptype)
    : PType(self_tag, 8, 8),
      element_ptype(element_ptype),
      elements_alignment(make_aligned(element_ptype->byte_size, element_ptype->alignment)),
      element_stride(make_aligned(element_ptype->byte_size, element_ptype->alignment)) {}
};

class PTuple : public PType {
  friend class TypeContext;
public:
  static const Tag self_tag = PType::Tag::TUPLE;
  const std::vector<const PType *> element_ptypes;
  const std::vector<size_t> element_offsets;
  PTuple(TypeContextToken,
	 std::vector<const PType *> element_ptypes,
	 std::vector<size_t> element_offsets,
	 size_t alignment,
	 size_t byte_size)
    : PType(self_tag, alignment, byte_size),
      element_ptypes(element_ptypes),
      element_offsets(element_offsets) {}
};

}

#endif
