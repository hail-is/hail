#include <llvm/IR/Type.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <hail/format.hpp>
#include <hail/query/backend/jit.hpp>
#include <hail/query/backend/stype.hpp>
#include <hail/query/backend/svalue.hpp>
#include <hail/query/ir.hpp>
#include <hail/query/ir_type.hpp>
#include <hail/type.hpp>
#include <hail/tunion.hpp>
#include <hail/vtype.hpp>
#include <hail/value.hpp>

namespace hail {

llvm::ExitOnError exit_on_error;

class Module;

class JITImpl {
  std::unique_ptr<llvm::orc::LLJIT> llvm_jit;

  const SType *stype_from(const VType *vtype);

public:
  JITImpl();

  JITModule compile(TypeContext &tc,
		    Module *m,
		    const std::vector<const VType *> &param_vtypes,
		    const VType *return_vtype);
};

JITImpl::JITImpl()
  : llvm_jit(std::move(exit_on_error(llvm::orc::LLJITBuilder().create()))) {}

class CompileModule {
private:
  TypeContext &tc;
  llvm::LLVMContext &llvm_context;
  llvm::Module *llvm_module;
public:
  CompileModule(TypeContext &tc,
		Module *module,
		const std::vector<const SType *> &param_stypes,
		const SType *return_stype,
		llvm::LLVMContext &llvm_context,
		llvm::Module *llvm_module);
};

class CompileFunction {
  TypeContext &tc;
  Function *function;
  llvm::LLVMContext &llvm_context;
  llvm::Module *llvm_module;

  /* Indexed by parameter index, the entry is the index of the first
     `llvm_function` parameter. */
  std::vector<size_t> param_llvm_start;
  
  llvm::Function *llvm_function;
  llvm::IRBuilder<> llvm_ir_builder;

  IRType ir_type;

  llvm::Type *get_llvm_type(PrimitiveType pt) const;

  llvm::AllocaInst *make_entry_alloca(llvm::Type *llvm_type);

  EmitValue emit(Block *x);
  EmitValue emit(Input *x);
  EmitValue emit(Literal *x);
  EmitValue emit(NA *x);
  EmitValue emit(IsNA *x);
  EmitValue emit(IR *x);

public:
  CompileFunction(TypeContext &tc,
		  Function *function,
		  const std::vector<const SType *> &param_types,
		  const SType *return_type,
		  llvm::LLVMContext &llvm_context,
		  llvm::Module *llvm_module);
};

CompileModule::CompileModule(TypeContext &tc,
			     Module *module,
			     const std::vector<const SType *> &param_stypes,
			     const SType *return_stype,
			     llvm::LLVMContext &llvm_context,
			     llvm::Module *llvm_module)
  : tc(tc),
    llvm_context(llvm_context),
    llvm_module(llvm_module) {
  auto main = module->get_function("main");
  // FIXME
  assert(main);

  CompileFunction(tc, main, param_stypes, return_stype, llvm_context, llvm_module);
}

CompileFunction::CompileFunction(TypeContext &tc,
				 Function *function,
				 const std::vector<const SType *> &param_stypes,
				 const SType *return_stype,
				 llvm::LLVMContext &llvm_context,
				 llvm::Module *llvm_module)
  : tc(tc),
    function(function),
    llvm_context(llvm_context),
    llvm_ir_builder(llvm_context),
    llvm_module(llvm_module),
    ir_type(tc, function) {
  std::vector<llvm::Type *> llvm_param_types;
  for (auto t : param_stypes) {
    param_llvm_start.push_back(llvm_param_types.size());
    for (auto pt : t->constituent_types())
      llvm_param_types.push_back(get_llvm_type(pt));
  }

  auto return_constituent_types = return_stype->constituent_types();
  auto llvm_return_type = get_llvm_type(return_constituent_types.size() == 1
					? return_constituent_types[0]
					: PrimitiveType::POINTER);

  auto llvm_ft = llvm::FunctionType::get(llvm_return_type, llvm_param_types, false);
  llvm_function = llvm::Function::Create(llvm_ft,
					 llvm::Function::ExternalLinkage,
					 "hl_compiled_main",
					 llvm_module);

  auto entry = llvm::BasicBlock::Create(llvm_context, "entry", llvm_function);
  llvm_ir_builder.SetInsertPoint(entry);

  emit(function->get_body());
}

llvm::Type *
CompileFunction::get_llvm_type(PrimitiveType pt) const {
  switch (pt) {
  case PrimitiveType::VOID:
    return llvm::Type::getVoidTy(llvm_context);
  case PrimitiveType::INT8:
    return llvm::Type::getInt8Ty(llvm_context);
  case PrimitiveType::INT32:
    return llvm::Type::getInt32Ty(llvm_context);
  case PrimitiveType::INT64:
    return llvm::Type::getInt64Ty(llvm_context);
  case PrimitiveType::FLOAT32:
    return llvm::Type::getFloatTy(llvm_context);
  case PrimitiveType::FLOAT64:
    return llvm::Type::getDoubleTy(llvm_context);
  case PrimitiveType::POINTER:
    return llvm::Type::getInt8PtrTy(llvm_context);
  default:
    abort();
  }
}

llvm::AllocaInst *
CompileFunction::make_entry_alloca(llvm::Type *llvm_type) {
  llvm::IRBuilder<> builder(&llvm_function->getEntryBlock(),
			    llvm_function->getEntryBlock().begin());
  return builder.CreateAlloca(llvm_type);
}

EmitValue
CompileFunction::emit(Block *x) {
  assert(x->get_children().size() == 1);
  return emit(x->get_child(0));
}

EmitValue
CompileFunction::emit(Input *x) {
  if (x->get_parent()->get_function_parent()) {
    size_t start = param_llvm_start[x->index];
    size_t end = (x->index + 1 < function->parameter_types.size()
		  ? function->parameter_types[x->index + 1]
		  : function->parameter_types.size());

    std::vector<Value *> param_values;
    for (size_t i = start; i < end; ++i)
      param_values.push_back(llvm_function->getArg(i));

    // FIXME vtypes should turn into emittypes, not stypes
    return param_stypes[x->index]->from_llvm_values(param_values);
  }

  abort();
}

EmitValue
CompileFunction::emit(Literal *x) {
  if (!x->value.get_present()) {
    auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, true));
    // FIXME nullptr
    return EmitValue(m, (const SValue *)nullptr);
  }

  switch (ir_type(x)->tag) {
  case Type::Tag::BOOL:
    {
      auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
      auto c = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, x->value.as_bool()));
      return EmitValue(m, new SBoolValue(new SBool(ir_type(x)), c));
    }
  case Type::Tag::INT32:
    {
      auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
      auto c = llvm::ConstantInt::get(llvm_context, llvm::APInt(32, x->value.as_int32()));
      return EmitValue(m, new SInt32Value(new SInt32(ir_type(x)), c));
    }
  case Type::Tag::INT64:
    {
      auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
      auto c = llvm::ConstantInt::get(llvm_context, llvm::APInt(64, x->value.as_int64()));
      return EmitValue(m, new SInt64Value(new SInt32(ir_type(x)), c));
    }
  case Type::Tag::FLOAT32:
    {
      auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
      auto c = llvm::ConstantFP::get(llvm_context, llvm::APFloat(x->value.as_float32()));
      return EmitValue(m, new SFloat32Value(new SFloat32(ir_type(x)), c));
    }
  case Type::Tag::FLOAT64:
    {
      auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
      auto c = llvm::ConstantFP::get(llvm_context, llvm::APFloat(x->value.as_float64()));
      return EmitValue(m, new SFloat64Value(new SFloat64(ir_type(x)), c));
    }
  default:
    abort();
  }
}

EmitValue
CompileFunction::emit(NA *x) {
  auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
  // FIXME nullptr
  return EmitValue(m, nullptr);
}

EmitValue
CompileFunction::emit(IsNA *x) {
  auto cond = emit(x->get_child(0)).as_control();

  llvm::AllocaInst *l = make_entry_alloca(llvm::Type::getInt8Ty(llvm_context));

  auto merge_bb = llvm::BasicBlock::Create(llvm_context, "isna_merge");
  llvm_ir_builder.SetInsertPoint(cond.missing_block);
  llvm_ir_builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(llvm_context), 1), l);
  llvm_ir_builder.CreateBr(merge_bb);

  llvm_ir_builder.SetInsertPoint(cond.present_block);
  llvm_ir_builder.CreateStore(llvm::ConstantInt::get(llvm::Type::getInt8Ty(llvm_context), 0), l);
  llvm_ir_builder.CreateBr(merge_bb);

  llvm_ir_builder.SetInsertPoint(merge_bb);
  llvm::Value *v = llvm_ir_builder.CreateLoad(l);

  auto m = llvm::ConstantInt::get(llvm_context, llvm::APInt(1, false));
  return EmitValue(m, new SBoolValue(new SBool(ir_type(x)), v));
}

EmitValue
CompileFunction::emit(IR *x) {
  return x->dispatch([this](auto x) {
		       return emit(x);
		     });
}

const SType *
JITImpl::stype_from(const VType *vtype) {
  switch (vtype->tag) {
  case VType::Tag::BOOL:
    return new SBool(vtype->type);
  case VType::Tag::INT32:
    return new SInt32(vtype->type);
  case VType::Tag::INT64:
    return new SInt64(vtype->type);
  case VType::Tag::FLOAT32:
    return new SFloat32(vtype->type);
  case VType::Tag::FLOAT64:
    return new SFloat64(vtype->type);
  default:
    abort();
  }
}

JITModule
JITImpl::compile(TypeContext &tc,
		 Module *module,
		 const std::vector<const VType *> &param_vtypes,
		 const VType *return_vtype) {
  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("hail", llvm_context);

  std::vector<const SType *> param_stypes;
  for (auto vt : param_vtypes)
    param_stypes.push_back(stype_from(vt));
  auto return_stype = stype_from(return_vtype);

  CompileModule mc(tc, module, param_stypes, return_stype, llvm_context, &llvm_module);

  uint64_t address = exit_on_error(llvm_jit->lookup("__hail_f")).getAddress();
  return JITModule(param_vtypes, return_vtype, address);

#if 0

  llvm::IRBuilder<> llvm_ir_builder(*llvm_context);

  auto llvm_m = std::make_unique<llvm::Module>("hail", *llvm_context);

  auto ft = llvm::FunctionType::get(llvm::Type::getInt32Ty(*llvm_context), std::vector<llvm::Type *>(), false);
  auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "__hail_f", llvm_m.get());

  auto bb = llvm::BasicBlock::Create(*llvm_context, "entry", f);
  llvm_ir_builder.SetInsertPoint(bb);

  llvm_ir_builder.CreateRet(llvm::ConstantInt::get(*llvm_context, llvm::APInt(32, -1)));

  verifyFunction(*f, &llvm::errs());

  llvm_m->print(llvm::errs(), nullptr);

  if (auto err = llvm_jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvm_m), std::move(llvm_context))))
    exit_on_error(std::move(err));

  uint64_t address = exit_on_error(llvm_jit->lookup("__hail_f")).getAddress();

  return JITModule(param_vtypes, return_vtype, address);
#endif
}

JITModule::JITModule(std::vector<const VType *> param_vtypes,
		     const VType *return_vtype,
		     uint64_t address)
  : param_vtypes(std::move(param_vtypes)),
    return_vtype(return_vtype),
    address(address) {}

JITModule::~JITModule() {}

Value
JITModule::invoke(ArenaAllocator &arena, const std::vector<Value> &args) {
  assert(param_vtypes.size() == args.size());
  for (size_t i = 0; i < param_vtypes.size(); ++i)
    assert(args[i].vtype == param_vtypes[i]);

  assert(isa<VBool>(param_vtypes[0])
	 && isa<VInt32>(param_vtypes[1])
	 && isa<VInt32>(return_vtype));

  auto fp = reinterpret_cast<uint32_t (*)(bool, int32_t)>(address);
  return Value(cast<VInt32>(return_vtype), (*fp)(args[0].as_bool(), args[1].as_int32()));
}

JIT::JIT() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  impl = std::make_unique<JITImpl>();
}

JIT::~JIT() {}

JITModule
JIT::compile(TypeContext &tc,
	     Module *m,
	     const std::vector<const VType *> &param_vtypes,
	     const VType *return_vtype) {
  // FIXME turn vtypes into stypes

  return std::move(impl->compile(tc, m, param_vtypes, return_vtype));
}

}
