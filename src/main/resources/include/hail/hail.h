#ifndef HAIL_HAIL_H
#define HAIL_HAIL_H 1
// hail.h - top-level header file for use by dynamically-generated C++ code
#include "hail/NativeObj.h"
#include "hail/NativeStatus.h"

// A dynamic-generated source file will be passed down as a Java String,
// together with some compiler options.  The combination of the compiler
// options and the string contents will be used to generate an 80-bit
// hash key, which will then be used both to generate the filename for the
// .cpp and .so (or .dylib) files.
//
// This hashcode will also be provided as -DHAIL_MODULE_KEY when compiling
// the code.

#define NAMESPACE_HAIL_MODULE_BEGIN \
  namespace hail { \
  namespace module_##HAIL_MODULE_KEY {

#define NAMESPACE_HAIL_MODULE_END \
  } }

#endif
