#ifndef HAIL_QUERY_BACKEND_JIT_HPP_INCLUDED
#define HAIL_QUERY_BACKEND_JIT_HPP_INCLUDED 1

#include <memory>

#include "hail/vtype.hpp"

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

  Value invoke(std::shared_ptr<ArenaAllocator> arena, const std::vector<Value> &arguments);
};

class JIT {
  std::unique_ptr<JITImpl> impl;

public:
  JIT();
  ~JIT();

  JITModule compile(HeapAllocator &heap,
		    TypeContext &tc,
		    Module *m,
		    const std::vector<const VType *> &param_vtypes,
		    const VType *return_vtype);
};

}

#endif
