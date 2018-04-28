//
// src/main/c/NativeObj.cpp
//
// Richard Cownie, Hail Team, 2018-04-26
//
#include "hail/NativeObj.h"

//
// There are complicated interactions between dynamic libraries, C++
// run-time type info, and vtables.  It turns out to be useful to
// have *some* non-inline methods to force creation of a single
// instance of the vtable.
//

NAMESPACE_BEGIN(hail)

#if 0
NativeObj::~NativeObj() { }

const char* NativeObj::getClassName() {
  return "NativeObj";
}

long NativeObj::getFieldOffset(int /*fieldSize*/, const char* /*fieldName*/) {
  return(-1);
}
#endif

NAMESPACE_END(hail)
