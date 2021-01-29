#ifndef HAIL_JIT_HPP_INCLUDED
#define HAIL_JIT_HPP_INCLUDED 1

#include <memory>

namespace hail {

class Module;
class JITImpl;

class JIT {
  std::unique_ptr<JITImpl> impl;

public:
  JIT();
  ~JIT();

  uint64_t compile(Module *m);
};

}

#endif
