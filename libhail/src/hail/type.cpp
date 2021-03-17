#include <hail/format.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>
#include <hail/vtype.hpp>
#include <tuple>

namespace hail {

Type::~Type() {}

TBlock::TBlock(TypeContextToken, std::vector<const Type *> input_types, std::vector<const Type *> output_types)
  : Type(Type::Tag::BLOCK),
    input_types(std::move(input_types)),
    output_types(std::move(output_types)) {}

void
format1(FormatStream &s, const Type *v) {
  switch (v->tag) {
  case Type::Tag::VOID:
    format(s, "void");
    break;
  case Type::Tag::BOOL:
    format(s, "bool");
    break;
  case Type::Tag::INT32:
    format(s, "int32");
    break;
  case Type::Tag::INT64:
    format(s, "int64");
    break;
  case Type::Tag::FLOAT32:
    format(s, "float32");
    break;
  case Type::Tag::FLOAT64:
    format(s, "float64");
    break;
  case Type::Tag::STR:
    format(s, "str");
    break;
  case Type::Tag::ARRAY:
    format(s, "array<", cast<TArray>(v)->element_type, ">");
    break;
  case Type::Tag::TUPLE:
    {
      format(s, "(");
      bool first = true;
      for (auto t : cast<TTuple>(v)->element_types) {
	if (first)
	  first = false;
	else
	  s.puts(", ");
	format(s, t);
      }
      format(s, ")");
    }
    break;
  default:
    abort();
  }
}

TypeContext::TypeContext(HeapAllocator &heap)
  : arena(heap),
    tvoid(arena.make<TVoid>(TypeContextToken())),
    tbool(arena.make<TBool>(TypeContextToken())),
    tint32(arena.make<TInt32>(TypeContextToken())),
    tint64(arena.make<TInt64>(TypeContextToken())),
    tfloat32(arena.make<TFloat32>(TypeContextToken())),
    tfloat64(arena.make<TFloat64>(TypeContextToken())),
    tstr(arena.make<TStr>(TypeContextToken())) {}

const TBlock *
TypeContext::tblock(const std::vector<const Type *> &input_types, const std::vector<const Type *> &output_types) {
  auto i = blocks.find(std::make_tuple(input_types, output_types));
  if (i != blocks.end())
    return i->second;

  TBlock *t = arena.make<TBlock>(TypeContextToken(), std::move(input_types), std::move(output_types));
  blocks.insert({std::make_tuple(t->input_types, t->output_types), t});
  return t;
}

const TArray *
TypeContext::tarray(const Type *element_type) {
  auto p = arrays.insert({element_type, nullptr});
  if (!p.second)
    return p.first->second;

  TArray *t = arena.make<TArray>(TypeContextToken(), element_type);
  p.first->second = t;
  return t;
}

const TStream *
TypeContext::tstream(const Type *element_type) {
  auto p = streams.insert({element_type, nullptr});
  if (!p.second)
    return p.first->second;

  TStream *t = arena.make<TStream>(TypeContextToken(), element_type);
  p.first->second = t;
  return t;
}

const TTuple *
TypeContext::ttuple(const std::vector<const Type *> &element_types) {
  /* it is not clear how to insert into tuples without copying
     element_types unnecessarily or searching tuples twice */
  auto i = tuples.find(element_types);
  if (i != tuples.end())
    return i->second;
  TTuple *t = arena.make<TTuple>(TypeContextToken(), element_types);
  tuples.insert({element_types, t});
  return t;
}

const VType *
TypeContext::get_vtype(const Type *type) {
  auto p = vtypes.insert({type, nullptr});
  if (!p.second)
    return p.first->second;

  const VType *vtype = nullptr;
  switch(type->tag) {
  case Type::Tag::BOOL:
    vtype = arena.make<VBool>(TypeContextToken(), cast<TBool>(type));
    break;
  case Type::Tag::INT32:
    vtype = arena.make<VInt32>(TypeContextToken(), cast<TInt32>(type));
    break;
  case Type::Tag::INT64:
    vtype = arena.make<VInt64>(TypeContextToken(), cast<TInt64>(type));
    break;
  case Type::Tag::FLOAT32:
    vtype = arena.make<VFloat32>(TypeContextToken(), cast<TFloat32>(type));
    break;
  case Type::Tag::FLOAT64:
    vtype = arena.make<VFloat64>(TypeContextToken(), cast<TFloat64>(type));
    break;
  case Type::Tag::STR:
    vtype = arena.make<VStr>(TypeContextToken(), cast<TStr>(type));
    break;
  case Type::Tag::ARRAY:
    {
      auto at = cast<TArray>(type);
      vtype = arena.make<VArray>(TypeContextToken(), at, get_vtype(at->element_type));
    }
    break;
  case Type::Tag::TUPLE:
    {
      auto tt = cast<TTuple>(type);
      std::vector<const VType *> element_vtypes;
      std::vector<size_t> element_offsets;
      size_t offset = 0;
      size_t alignment = 1;
      for (auto element_type : tt->element_types) {
	auto element_vtype = get_vtype(element_type);
	element_vtypes.push_back(element_vtype);
	offset = make_aligned(offset, element_vtype->alignment);
	element_offsets.push_back(offset);
	offset += element_vtype->byte_size;
	alignment = std::max(alignment, element_vtype->alignment);
      }
      vtype = arena.make<VTuple>(TypeContextToken(), tt, element_vtypes, element_offsets, alignment, offset);
    }
    break;
  default:
    abort();
  }
  p.first->second = vtype;
  return vtype;
}

}
