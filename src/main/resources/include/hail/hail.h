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
// This hashcode will also be provided as part of -DHAIL_MODULE when
// compiling the code.

#define NAMESPACE_HAIL_MODULE_BEGIN \
  namespace hail { \
  namespace HAIL_MODULE {

#define NAMESPACE_HAIL_MODULE_END \
  } }

template<typename T>
inline ssize_t ssize(const T& container) { return static_cast<ssize_t>(container.size()); }

template<>
inline ssize_t ssize<size_t>(const size_t& value) { return static_cast<ssize_t>(value); }

#endif
