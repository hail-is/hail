#ifndef HAIL_TUNION_HPP
#define HAIL_TUNION_HPP 1

#include <cassert>

namespace hail {

template<class T> bool
isa(const typename T::BaseType *p) {
  return T::is_instance_tag(p->tag);
}

template<class T> T *
cast(typename T::BaseType *p) {
  assert(isa<T>(p));
  return static_cast<T *>(p); 
}

template<class T> const T *
cast(const typename T::BaseType *p) {
  assert(isa<T>(p));
  return static_cast<const T *>(p);
}

template<class T> T *
dyn_cast(typename T::BaseType *p) {
  if (isa<T>(p))
    return static_cast<T *>(p);
  else
    return nullptr;
}

template<class T> const T *
dyn_cast(const typename T::BaseType *p) {
  if (isa<T>(p))
    return static_cast<const T *>(p);
  else
    return nullptr;
}

}

#endif
