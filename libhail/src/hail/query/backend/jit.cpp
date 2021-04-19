#include <llvm/IR/Type.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include "hail/format.hpp"
#include "hail/query/backend/compile.hpp"
#include "hail/query/backend/jit.hpp"
#include "hail/query/backend/stype.hpp"
#include "hail/query/backend/svalue.hpp"
#include "hail/query/ir.hpp"
#include "hail/query/ir_type.hpp"
#include "hail/runtime/runtime.hpp"
#include "hail/type.hpp"
#include "hail/tunion.hpp"
#include "hail/vtype.hpp"
#include "hail/value.hpp"

namespace hail {

class JITImpl {
  llvm::ExitOnError exit_on_error;
  std::unique_ptr<llvm::orc::LLJIT> llvm_jit;

  const SType *stype_from(const VType *vtype);
  EmitType emit_type_from(const VType *vtype);

public:
  JITImpl();

  JITModule compile(HeapAllocator &heap,
		    TypeContext &tc,
		    Module *module,
		    const std::vector<const VType *> &param_vtypes,
		    const VType *return_vtype);
};

JITImpl::JITImpl()
  : llvm_jit(std::move(exit_on_error(llvm::orc::LLJITBuilder().create()))) {
  auto &llvm_data_layout = llvm_jit->getDataLayout();
  llvm::orc::MangleAndInterner mangle(llvm_jit->getExecutionSession(), llvm_data_layout);
  auto &jit_dylib = llvm_jit->getMainJITDylib();
  auto sym = llvm::orc::absoluteSymbols({
					 {mangle("hl_runtime_region_allocate"),
					  llvm::JITEvaluatedSymbol(llvm::pointerToJITTargetAddress(&hl_runtime_region_allocate),
								   llvm::JITSymbolFlags::Exported)}
    });
  if (auto err = jit_dylib.define(sym))
    exit_on_error(std::move(err));
}

JITModule
JITImpl::compile(HeapAllocator &heap,
		 TypeContext &tc,
		 Module *module,
		 const std::vector<const VType *> &param_vtypes,
		 const VType *return_vtype) {
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = std::make_unique<llvm::Module>("hail", *llvm_context);

  STypeContext stc(heap, tc);
  std::vector<EmitType> param_types;
  for (auto vt : param_vtypes)
    param_types.push_back(stc.emit_type_from(vt));
  auto return_type = stc.emit_type_from(return_vtype);

  CompileModule mc(tc, stc, module, param_types, return_type, *llvm_context, llvm_module.get());

  if (auto err = llvm_jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvm_module), std::move(llvm_context))))
    exit_on_error(std::move(err));

  uint64_t address = exit_on_error(llvm_jit->lookup("__hail_f")).getAddress();
  return JITModule(param_vtypes, return_vtype, address);
}

JITModule::JITModule(std::vector<const VType *> param_vtypes,
		     const VType *return_vtype,
		     uint64_t address)
  : param_vtypes(std::move(param_vtypes)),
    return_vtype(return_vtype),
    address(address) {}

JITModule::~JITModule() {}

Value
JITModule::invoke(std::shared_ptr<ArenaAllocator> region, const std::vector<Value> &args) {
  assert(param_vtypes.size() == args.size());
  for (size_t i = 0; i < param_vtypes.size(); ++i)
    assert(args[i].vtype == param_vtypes[i]);

  Value::Raw raw_args[args.size()];
  for (size_t i = 0; i < args.size(); ++i)
    raw_args[i] = args[i];
  Value::Raw raw_return_value;

  reinterpret_cast<void (*)(char *, char *, char *)>
    (address)(reinterpret_cast<char *>(&region->raw_arena),
	      reinterpret_cast<char *>(&raw_return_value),
	      reinterpret_cast<char *>(&raw_args));

  return Value(return_vtype, std::move(region), raw_return_value);
}

JIT::JIT() {
  // FIXME only do once
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  impl = std::make_unique<JITImpl>();
}

JIT::~JIT() {}

JITModule
JIT::compile(HeapAllocator &heap,
	     TypeContext &tc,
	     Module *m,
	     const std::vector<const VType *> &param_vtypes,
	     const VType *return_vtype) {
  // FIXME turn vtypes into emit types
  // FIXME then call impl with emit types
  return std::move(impl->compile(heap, tc, m, param_vtypes, return_vtype));
}

}
