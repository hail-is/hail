#ifndef HAIL_PARTITIONITERATORS_H
#define HAIL_PARTITIONITERATORS_H 1
#include "hail/Upcalls.h"
#include "hail/Utils.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <memory>
#include <cstring>

namespace hail {

class JavaIteratorWrapper : public NativeObj {
  private:
    UpcallEnv up_;
    jobject jrvit_;
    char * row_;

  public:
    JavaIteratorWrapper() = delete;
    JavaIteratorWrapper(JavaIteratorWrapper &it);
    JavaIteratorWrapper(JavaIteratorWrapper &&it);
    JavaIteratorWrapper(UpcallEnv up, jobject jrvit);
    bool advance();
    char const* get();
    ~JavaIteratorWrapper();
};

class JavaRVIterator {
  private:
    JavaIteratorWrapper * jit_;

  public:
    JavaRVIterator() = delete;
    JavaRVIterator(JavaIteratorWrapper * jrvit);
    JavaRVIterator& operator++();
    friend bool operator==(JavaRVIterator const& lhs, JavaRVIterator const& rhs) {
      return (lhs.jit_ == rhs.jit_);
    }
    friend bool operator!=(JavaRVIterator const& lhs, JavaRVIterator const& rhs) {
      return !(lhs == rhs);
    }
    char const* operator*();
};

JavaRVIterator begin(JavaIteratorWrapper * it) { return JavaRVIterator(it); };
JavaRVIterator end(JavaIteratorWrapper * it) { return JavaRVIterator(nullptr); };

}

#endif