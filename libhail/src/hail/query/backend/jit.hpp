#ifndef HAIL_JIT_HPP_INCLUDED
#define HAIL_JIT_HPP_INCLUDED 1

#include <hail/vtype.hpp>
#include <memory>

namespace hail {

class Value;
class Module;
class JITImpl;

class JITModule {
  friend class JITImpl;
public:
  const std::vector<const VType *> param_vtypes;
  const VType *const return_vtype;
private:
  uint64_t address;

  JITModule(std::vector<const VType *> param_vtypes,
	    const VType *return_vtype,
	    uint64_t address);
public:
  ~JITModule();

  Value invoke(ArenaAllocator &arena, const std::vector<Value> &arguments);
};

class JIT {
  std::unique_ptr<JITImpl> impl;

public:
  JIT();
  ~JIT();

  JITModule compile(Module *m, const std::vector<const VType *> &param_vtypes, const VType *return_vtype);
};

}

#endif
