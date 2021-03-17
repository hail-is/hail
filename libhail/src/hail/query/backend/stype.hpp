#ifndef HAIL_QUERY_BACKEND_STYPE_HPP_INCLUDED
#define HAIL_QUERY_BACKEND_STYPE_HPP_INCLUDED 1

#include <llvm/IR/Value.h>
#include <hail/hash.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>
#include <vector>

namespace hail {

class EmitType;
class SValue;
class EmitValue;
class CompileFunction;
class MemorySize;
class VTuple;

enum class PrimitiveType {
  VOID,
  INT8,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
  POINTER
};

class STypeContextToken {
  friend class STypeContext;
  STypeContextToken() {}
};

class SType {
public:
  using BaseType = SType;
  enum class Tag {
    BOOL,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    CANONICALTUPLE,
    STACKTUPLE
  };
  const Tag tag;
  const Type *const type;
  SType(Tag tag, const Type *type) : tag(tag), type(type) {}
  virtual ~SType();

  // FIXME make return value an iterator?  A generator!
  void get_constituent_types(std::vector<PrimitiveType> &constituent_types) const;

  SValue *from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) const;

  const SValue *load_from_address(CompileFunction &cf, llvm::Value *address) const;

  const SValue *construct_from_value(CompileFunction &cf, const SValue *from) const;

  void construct_at_address_from_value(CompileFunction &cf, llvm::Value *address, const SValue *from) const;

  MemorySize get_memory_size() const;

  template<typename F> auto dispatch(F f) const;
};

class MemorySize {
public:
  const size_t byte_size;
  const size_t alignment;

public:
  MemorySize(size_t byte_size, size_t alignment)
    : byte_size(byte_size), alignment(alignment) {}
};

template<typename T> inline constexpr bool
is_heap_stype() {
  static_assert(std::is_base_of_v<SType, T>);
  return std::is_base_of_v<MemorySize, T>;
}

class SBool : public SType, public MemorySize {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::BOOL; }
  SBool(STypeContextToken, const Type *type);
};

class SInt32 : public SType, public MemorySize {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::INT32; }
  SInt32(STypeContextToken, const Type *type);
};

class SInt64 : public SType, public MemorySize {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::INT64; }
  SInt64(STypeContextToken, const Type *type);
};

class SFloat32 : public SType, public MemorySize {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::FLOAT32; }
  SFloat32(STypeContextToken, const Type *type);
};

class SFloat64 : public SType, public MemorySize {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::FLOAT64; }
  SFloat64(STypeContextToken, const Type *type);
};

class SCanonicalTuple : public SType, public MemorySize {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::CANONICALTUPLE; }
  std::vector<EmitType> element_types;
  std::vector<size_t> element_offsets;
  SCanonicalTuple(STypeContextToken,
		  const Type *type,
		  std::vector<EmitType> element_types,
		  size_t byte_size,
		  size_t alignment,
		  std::vector<size_t> element_offsets);

  void set_element_missing(CompileFunction &cf, llvm::Value *address, size_t i, bool missing) const;
};

class SStackTuple : public SType {
public:
  static bool is_instance_tag(Tag tag) { return tag == SType::Tag::STACKTUPLE; }
  std::vector<EmitType> element_types;
  SStackTuple(STypeContextToken, const Type *type, std::vector<EmitType> element_types);
};

class EmitType {
public:
  const SType *const stype;

  EmitType(const SType *stype) : stype(stype) {}

  bool operator==(EmitType that) const { return stype == that.stype; }
  bool operator!=(EmitType that) const { return stype != that.stype; }

  void get_constituent_types(std::vector<PrimitiveType> &constituent_types) const;

  EmitValue from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) const;

  EmitValue make_na(CompileFunction &cf) const;
};

}

namespace std {

template<>
struct hash<hail::EmitType> {
  size_t operator()(hail::EmitType emit_type) const {
    return hash_value(emit_type.stype);
  }
};

}

namespace hail {

class STypeContext {
  TypeContext &tc;
  ArenaAllocator arena;

  std::unordered_map<std::vector<EmitType>, const SCanonicalTuple *> canonical_stuples;
  std::unordered_map<std::vector<EmitType>, const SStackTuple *> stack_stuples;

public:
  const SBool *const sbool;
  const SInt32 *const sint32;
  const SInt64 *const sint64;
  const SFloat32 *const sfloat32;
  const SFloat64 *const sfloat64;

  STypeContext(HeapAllocator &heap, TypeContext &tc);

  const SCanonicalTuple *canonical_stuple(const Type *type, const std::vector<EmitType> &element_types, const VTuple *vtuple);
  const SStackTuple *stack_stuple(const Type *type, const std::vector<EmitType> &element_types);

  EmitType emit_type_from(const VType *vtype);

  const SType *stype_from(const VType *vtype);
};

template<typename F> auto
SType::dispatch(F f) const {
  switch(tag) {
  case Tag::BOOL: return f(cast<SBool>(this));
  case Tag::INT32: return f(cast<SInt32>(this));
  case Tag::INT64: return f(cast<SInt64>(this));
  case Tag::FLOAT32: return f(cast<SFloat32>(this));
  case Tag::FLOAT64: return f(cast<SFloat64>(this));
  case Tag::CANONICALTUPLE: return f(cast<SCanonicalTuple>(this));
  case Tag::STACKTUPLE: return f(cast<SStackTuple>(this));
  default:
    abort();
  }
}

}

#endif
