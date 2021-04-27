#include <llvm/IR/Constants.h>

#include "hail/query/backend/compile.hpp"
#include "hail/query/backend/stype.hpp"
#include "hail/query/backend/svalue.hpp"

namespace hail {

SType::~SType() {}

void
SType::get_constituent_types(std::vector<PrimitiveType> &constituent_types) const {
  switch (tag) {
  case SType::Tag::BOOL:
    constituent_types.push_back(PrimitiveType::INT8);
    break;
  case SType::Tag::INT32:
    constituent_types.push_back(PrimitiveType::INT32);
    break;
  case SType::Tag::INT64:
    constituent_types.push_back(PrimitiveType::INT64);
    break;
  case SType::Tag::FLOAT32:
    constituent_types.push_back(PrimitiveType::FLOAT32);
    break;
  case SType::Tag::FLOAT64:
    constituent_types.push_back(PrimitiveType::FLOAT64);
    break;
  case SType::Tag::CANONICALTUPLE:
    constituent_types.push_back(PrimitiveType::POINTER);
    break;
  default:
    abort();
  }
}

SValue *
SType::from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) const {
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
  case SType::Tag::CANONICALTUPLE:
    return new SCanonicalTupleValue(this, llvm_values[i]);
  default:
    abort();
  }
}

const SValue *
SType::load_from_address(CompileFunction &cf, llvm::Value *address) const {
  switch (tag) {
  case SType::Tag::BOOL:
    {
      auto v = cf.llvm_ir_builder.CreateLoad(address);
      return new SBoolValue(this, v);
    }
  case SType::Tag::INT32:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getInt32Ty(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SInt32Value(this, v);
    }
  case SType::Tag::INT64:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getInt64Ty(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SInt64Value(this, v);
    }
  case SType::Tag::FLOAT32:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getFloatTy(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SFloat32Value(this, v);
    }
  case SType::Tag::FLOAT64:
    {
      auto typed_address = cf.llvm_ir_builder.CreateBitCast(address,
							    llvm::PointerType::get(llvm::Type::getDoubleTy(cf.llvm_context), 0));
      auto v = cf.llvm_ir_builder.CreateLoad(typed_address);
      return new SFloat64Value(this, v);
    }
  case SType::Tag::CANONICALTUPLE:
    {
      return new SCanonicalTupleValue(this, address);
    }
  default:
    abort();
  }
}

const SValue *
SType::construct_from_value(CompileFunction &cf, const SValue *from) const {
  if (from->stype == this)
    return from;

  auto runtime_allocate_ft =
    llvm::FunctionType::get(llvm::Type::getInt8PtrTy(cf.llvm_context),
			    {llvm::Type::getInt8PtrTy(cf.llvm_context),
			     llvm::Type::getInt64Ty(cf.llvm_context),
			     llvm::Type::getInt64Ty(cf.llvm_context)},
			    false);
  auto runtime_allocate_f = llvm::Function::Create(runtime_allocate_ft,
						   llvm::Function::ExternalLinkage,
						   "hl_runtime_region_allocate",
						   cf.llvm_module);

  auto memory_size = get_memory_size();
  auto address = cf.llvm_ir_builder.CreateCall(runtime_allocate_f,
					       {cf.llvm_function->getArg(0),
						llvm::ConstantInt::get(cf.llvm_context, llvm::APInt(64, memory_size.byte_size)),
						llvm::ConstantInt::get(cf.llvm_context, llvm::APInt(64, memory_size.alignment))});
  construct_at_address_from_value(cf, address, from);
  return load_from_address(cf, address);
}

void
SType::construct_at_address_from_value(CompileFunction &cf, llvm::Value *address, const SValue *from) const {
  switch (tag) {
  case SType::Tag::BOOL:
    cf.llvm_ir_builder.CreateStore(cast<SBoolValue>(from)->value, address);
    break;
  case SType::Tag::INT32:
    cf.llvm_ir_builder.CreateStore(cast<SInt32Value>(from)->value,
				   cf.llvm_ir_builder.CreateBitCast(address,
								    llvm::PointerType::get(llvm::Type::getInt32Ty(cf.llvm_context), 0)));
    break;
  case SType::Tag::INT64:
    cf.llvm_ir_builder.CreateStore(cast<SInt64Value>(from)->value,
				   cf.llvm_ir_builder.CreateBitCast(address,
								    llvm::PointerType::get(llvm::Type::getInt64Ty(cf.llvm_context), 0)));
    break;
  case SType::Tag::FLOAT32:
    cf.llvm_ir_builder.CreateStore(cast<SFloat32Value>(from)->value,
				   cf.llvm_ir_builder.CreateBitCast(address,
								    llvm::PointerType::get(llvm::Type::getFloatTy(cf.llvm_context), 0)));
    break;
  case SType::Tag::FLOAT64:
    cf.llvm_ir_builder.CreateStore(cast<SFloat64Value>(from)->value,
				   cf.llvm_ir_builder.CreateBitCast(address,
								    llvm::PointerType::get(llvm::Type::getDoubleTy(cf.llvm_context), 0)));
    break;
  case SType::Tag::CANONICALTUPLE:
    {
      auto t = cast<SCanonicalTuple>(this);
      auto v = cast<STupleValue>(from);
      for (size_t i = 0; i < t->element_offsets.size(); ++i) {
	auto c = v->get_element(cf, i).as_control(cf);

	auto merge_block = llvm::BasicBlock::Create(cf.llvm_context, "make_tuple_after_element", cf.llvm_function);

	auto element_address = cf.llvm_ir_builder.CreateGEP(address,
							    llvm::ConstantInt::get(cf.llvm_context, llvm::APInt(64, t->element_offsets[i])));

	t->set_element_missing(cf, address, i, false);
	t->element_types[i].stype->construct_at_address_from_value(cf, element_address, c.svalue);
	cf.llvm_ir_builder.CreateBr(merge_block);

	cf.llvm_ir_builder.SetInsertPoint(c.missing_block);
	t->set_element_missing(cf, address, i, true);
	cf.llvm_ir_builder.CreateBr(merge_block);

	cf.llvm_ir_builder.SetInsertPoint(merge_block);
      }
    }
    break;
  default:
    abort();
  }
}

MemorySize
SType::get_memory_size() const {
  return dispatch([](auto t) -> MemorySize {
		    if constexpr (std::is_convertible_v<decltype(*t), MemorySize>)
		      return MemorySize(*t);
		    abort();
		  });
}

SBool::SBool(STypeContextToken, const Type *type)
  : SType(Tag::BOOL, type), MemorySize(1, 1) {}

SInt32::SInt32(STypeContextToken, const Type *type)
  : SType(Tag::INT32, type), MemorySize(4, 4) {}

SInt64::SInt64(STypeContextToken, const Type *type)
  : SType(Tag::INT64, type), MemorySize(8, 8) {}

SFloat32::SFloat32(STypeContextToken, const Type *type)
  : SType(Tag::FLOAT32, type), MemorySize(4, 4) {}

SFloat64::SFloat64(STypeContextToken, const Type *type)
  : SType(Tag::FLOAT64, type), MemorySize(8, 8) {}

SCanonicalTuple::SCanonicalTuple(STypeContextToken,
				 const Type *type,
				 std::vector<EmitType> element_types,
				 size_t byte_size,
				 size_t alignment,
				 std::vector<size_t> element_offsets)
  : SType(Tag::CANONICALTUPLE, type),
    MemorySize(byte_size, alignment),
    element_types(std::move(element_types)),
    element_offsets(std::move(element_offsets)) {}

void
SCanonicalTuple::set_element_missing(CompileFunction &cf, llvm::Value *address, size_t i, bool missing) const {
  llvm::Value *missing_address =
    cf.llvm_ir_builder.CreateGEP(address,
				 llvm::ConstantInt::get(cf.llvm_context, llvm::APInt(64, i >> 3)));
  llvm::Value *b = cf.llvm_ir_builder.CreateLoad(missing_address);
  if (missing)
    b = cf.llvm_ir_builder.CreateOr(b,
				    llvm::ConstantInt::get(cf.llvm_context, llvm::APInt(8, 1 << (i & 7))));
  else
    b = cf.llvm_ir_builder.CreateAnd(b,
				     llvm::ConstantInt::get(cf.llvm_context, llvm::APInt(8, ~(1 << (i & 7)))));
  cf.llvm_ir_builder.CreateStore(b, missing_address);
}

SStackTuple::SStackTuple(STypeContextToken, const Type *type, std::vector<EmitType> element_types)
  : SType(Tag::STACKTUPLE, type),
    element_types(std::move(element_types)) {}

void
EmitType::get_constituent_types(std::vector<PrimitiveType> &constituent_types) const {
  constituent_types.push_back(PrimitiveType::INT8);
  stype->get_constituent_types(constituent_types);
}

EmitValue
EmitType::from_llvm_values(const std::vector<llvm::Value *> &llvm_values, size_t i) const {
  llvm::Value *missing = llvm_values[i++];
  SValue *svalue = stype->from_llvm_values(llvm_values, i);
  return EmitValue(missing, svalue);
}

EmitValue
EmitType::make_na(CompileFunction &cf) const {
  auto m = llvm::ConstantInt::get(llvm::Type::getInt8Ty(cf.llvm_context), 1);

  std::vector<llvm::Value *> llvm_values;
  std::vector<PrimitiveType> constituent_types;
  stype->get_constituent_types(constituent_types);
  for (auto pt : constituent_types)
    llvm_values.push_back(llvm::UndefValue::get(cf.get_llvm_type(pt)));
  auto svalue = stype->from_llvm_values(llvm_values, 0);

  return EmitValue(m, svalue);
}

STypeContext::STypeContext(HeapAllocator &heap, TypeContext &tc)
  : tc(tc),
    arena(heap),
    sbool(arena.make<SBool>(STypeContextToken(), tc.tbool)),
    sint32(arena.make<SInt32>(STypeContextToken(), tc.tint32)),
    sint64(arena.make<SInt64>(STypeContextToken(), tc.tint64)),
    sfloat32(arena.make<SFloat32>(STypeContextToken(), tc.tfloat32)),
    sfloat64(arena.make<SFloat64>(STypeContextToken(), tc.tfloat64)) {}

const SCanonicalTuple *
STypeContext::canonical_stuple(const Type *type, const std::vector<EmitType> &element_types, const VTuple *vtuple) {
  // FIXME offsets not hashed
  auto p = canonical_stuples.insert({element_types, nullptr});
  if (!p.second)
    return p.first->second;

  const SCanonicalTuple *t = arena.make<SCanonicalTuple>(STypeContextToken(), type, element_types,
							 vtuple->byte_size,
							 vtuple->alignment,
							 vtuple->element_offsets);
  p.first->second = t;
  return t;
}

const SStackTuple *
STypeContext::stack_stuple(const Type *type, const std::vector<EmitType> &element_types) {
  auto p = stack_stuples.insert({element_types, nullptr});
  if (!p.second)
    return p.first->second;

  const SStackTuple *t = arena.make<SStackTuple>(STypeContextToken(), type, element_types);
  p.first->second = t;
  return t;
}

EmitType
STypeContext::emit_type_from(const VType *vtype) {
  return EmitType(stype_from(vtype));
}

const SType *
STypeContext::stype_from(const VType *vtype) {
  switch (vtype->tag) {
  case VType::Tag::BOOL:
    return sbool;
  case VType::Tag::INT32:
    return sint32;
  case VType::Tag::INT64:
    return sint64;
  case VType::Tag::FLOAT32:
    return sfloat32;
  case VType::Tag::FLOAT64:
    return sfloat64;
  case VType::Tag::TUPLE:
    {
      const VTuple *vt = cast<VTuple>(vtype);
      std::vector<EmitType> element_types;
      for (auto et : vt->element_vtypes)
	element_types.push_back(emit_type_from(et));
      return canonical_stuple(vt->type, element_types, vt);
    }
  default:
    abort();
  }
}

}
