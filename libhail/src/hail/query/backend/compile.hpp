#ifndef HAIL_QUERY_BACKEND_COMPILE_HPP_INCLUDED
#define HAIL_QUERY_BACKEND_COMPILE_HPP_INCLUDED 1

#include <llvm/IR/IRBuilder.h>

#include "hail/query/backend/stype.hpp"
#include "hail/query/ir_type.hpp"

namespace llvm {

class LLVMContext;
class Type;
class Module;
class Function;
class AllocaInst;

}

namespace hail {

class TypeContext;
class Module;
class Function;

class CompileModule {
public:
  TypeContext &tc;
  STypeContext &stc;
  llvm::LLVMContext &llvm_context;
  llvm::Module *llvm_module;

  llvm::Value* region_allocate(llvm::IRBuilder<> *llvm_ir_builder, llvm::Value *region, MemorySize *memory_size);
  llvm::Value* region_allocate(llvm::IRBuilder<> *llvm_ir_builder, llvm::Value *region, llvm::Value *alignment, llvm::Value *num_bytes);
  void print_float64(llvm::IRBuilder<> *llvm_ir_builder, llvm::Value *double_to_print);
  void print_bool(llvm::IRBuilder<> *llvm_ir_builder, llvm::Value *bool_to_print);
  void printf(llvm::IRBuilder<> *llvm_ir_builder, const char* string, std::vector<llvm::Value *> args);

  CompileModule(TypeContext &tc,
		STypeContext &stc,
		Module *module,
		const std::vector<EmitType> &param_types,
		EmitType return_type,
		llvm::LLVMContext &llvm_context,
		llvm::Module *llvm_module);

private:
  llvm::Function *runtime_allocate_f;
  llvm::Function *runtime_printf;
};

class CompileFunction {
public:
  TypeContext &tc;
  STypeContext &stc;
  Function *function;
  const std::vector<EmitType> &param_types;
  EmitType return_type;
  llvm::LLVMContext &llvm_context;
  CompileModule *module;

  /* Indexed by parameter index, the entry is the index of the first
     `llvm_function` parameter. */
  std::vector<size_t> param_llvm_start;

  llvm::Function *llvm_function;
  // FIXME rename llvm_builder
  llvm::IRBuilder<> llvm_ir_builder;

  IRType ir_type;

  // FIXME move to SType
  const SType *get_default_stype(const Type *t);

  llvm::Type *get_llvm_type(PrimitiveType pt) const;
  llvm::AllocaInst *make_entry_alloca(llvm::Type *llvm_type);

  EmitValue emit(Block *x);
  EmitValue emit(Input *x);
  EmitValue emit(Literal *x);
  EmitValue emit(NA *x);
  EmitValue emit(IsNA *x);
  EmitValue emit(MakeTuple *x);
  EmitValue emit(GetTupleElement *x);
  EmitValue emit(MakeArray *x);
  EmitValue emit(ArrayLen *x);
  EmitValue emit(ArrayRef *x);
  EmitValue emit(IR *x);

  CompileFunction(TypeContext &tc,
		  STypeContext &stc,
		  Function *function,
		  const std::vector<EmitType> &param_types,
		  EmitType return_type,
		  llvm::LLVMContext &llvm_context,
		  CompileModule *module);
};

}

#endif
