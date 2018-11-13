#ifndef HAIL_PARTITIONITERATORS_H
#define HAIL_PARTITIONITERATORS_H 1
#include "hail/Upcalls.h"
#include "hail/Utils.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <memory>
#include <cstring>

namespace hail {

class JavaIteratorObject : public NativeObj {
  private:
    UpcallEnv up_;
    jobject jrvit_;
    char * row_;
    bool advance();
    char const* get() const;

  public:
    JavaIteratorObject() = delete;
    JavaIteratorObject(JavaIteratorObject &it);
    JavaIteratorObject(JavaIteratorObject &&it);
    JavaIteratorObject(UpcallEnv up, jobject jrvit);
    ~JavaIteratorObject();

    class RVIterator {
      friend class JavaIteratorObject;
      private:
        JavaIteratorObject * jit_;
        explicit RVIterator(JavaIteratorObject * jrvit);

      public:
        RVIterator() = delete;
        RVIterator& operator++();
        friend bool operator==(RVIterator const& lhs, RVIterator const& rhs) {
          return (lhs.jit_ == rhs.jit_);
        }
        friend bool operator!=(RVIterator const& lhs, RVIterator const& rhs) {
          return !(lhs == rhs);
        }
        char const* operator*() const;
    };
    RVIterator begin() { return RVIterator(this); };
    RVIterator end() { return RVIterator(nullptr); };
};

using RVIterator = JavaIteratorObject::RVIterator;

}

#endif