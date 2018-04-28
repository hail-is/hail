#ifndef HAIL_COMMONDEFS_H
#define HAIL_COMMONDEFS_H 1
//
// A few useful macros
//

// A way to begin/end namespaces without making a text editor change your
// indentation level.

#define NAMESPACE_BEGIN(_name) namespace _name {
#define NAMESPACE_END(_name)   }

#define NAMESPACE_BEGIN_ANON   namespace {
#define NAMESPACE_END_ANON     }

#define NAMESPACE_BEGIN_MODULE namespace module_##HAIL_MODULE_KEY {
#define NAMESPACE_END_MODULE   }


// Declare a function to be exported from a DLL with "C" linkage

#define NATIVEMETHOD(cppReturnType, scalaClass, scalaMethod) \
  extern "C" __attribute__((visibility("default"))) \
    cppReturnType Java_is_hail_nativecode_##scalaClass##_##scalaMethod

#endif
