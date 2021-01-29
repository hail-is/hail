#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <hail/jit.hpp>
#include <hail/format.hpp>
#include <hail/type.hpp>
#include <hail/ir.hpp>

namespace hail {

class Module;

class JITImpl {
  llvm::ExitOnError exit_on_error;
  std::unique_ptr<llvm::orc::LLJIT> llvm_jit;

public:
  JITImpl();

  uint64_t compile(Module *m);
};

JITImpl::JITImpl()
  : llvm_jit(std::move(exit_on_error(llvm::orc::LLJITBuilder().create()))) {}

class JITModuleContext {
  llvm::ExitOnError exit_on_error;
  std::unique_ptr<llvm::LLVMContext> llvm_context;
  llvm::IRBuilder<> llvm_ir_builder;
   std::unique_ptr<llvm::Module> llvm_module;

  std::map<const Type *, llvm::Type *> llvm_types;

  void compile(Function *f);

public:
  JITModuleContext();

  llvm::orc::ThreadSafeModule compile(Module *m);
};

JITModuleContext::JITModuleContext()
  : llvm_context(std::make_unique<llvm::LLVMContext>()),
    llvm_ir_builder(*llvm_context),
    llvm_module(std::make_unique<llvm::Module>("hail", *llvm_context)) {}

void
JITModuleContext::compile(Function *f) {
  // FIXME we need STypes to write this
  abort();
}

// FIXME this needs to take a set of emittypes for parameters
// and return an stype for the result.
llvm::orc::ThreadSafeModule
JITModuleContext::compile(Module *m) {
  // FIXME
#if 0
  auto main = m->get_function("main");
  if (!main)
    ???

  compile(f);
#endif
  abort();

  return llvm::orc::ThreadSafeModule(std::move(llvm_module), std::move(llvm_context));
}

uint64_t
JITImpl::compile(Module *m) {
  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  llvm::IRBuilder<> llvm_ir_builder(*llvm_context);

  auto llvm_m = std::make_unique<llvm::Module>("hail", *llvm_context);

  auto ft = llvm::FunctionType::get(llvm::Type::getInt32Ty(*llvm_context), std::vector<llvm::Type *>(), false);
  auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "__hail_f", llvm_m.get());

  auto bb = llvm::BasicBlock::Create(*llvm_context, "entry", f);
  llvm_ir_builder.SetInsertPoint(bb);

  llvm_ir_builder.CreateRet(llvm::ConstantInt::get(*llvm_context, llvm::APInt(32, 357)));

  verifyFunction(*f, &llvm::errs());

  llvm_m->print(llvm::errs(), nullptr);

  if (auto err = llvm_jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvm_m), std::move(llvm_context))))
    exit_on_error(std::move(err));

  return exit_on_error(llvm_jit->lookup("__hail_f")).getAddress();
}

JIT::JIT() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  impl = std::make_unique<JITImpl>();
}

JIT::~JIT() {}

uint64_t
JIT::compile(Module *m) {
  return impl->compile(m);
}

}
