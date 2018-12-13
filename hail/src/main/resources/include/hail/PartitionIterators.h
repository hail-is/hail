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

    class Iterator {
      friend class JavaIteratorObject;
      private:
        JavaIteratorObject * jit_;
        explicit Iterator(JavaIteratorObject * jrvit);

      public:
        Iterator() = delete;
        Iterator& operator++();
        friend bool operator==(Iterator const& lhs, Iterator const& rhs) {
          return (lhs.jit_ == rhs.jit_);
        }
        friend bool operator!=(Iterator const& lhs, Iterator const& rhs) {
          return !(lhs == rhs);
        }
        char const* operator*() const;
    };
    Iterator begin() { return Iterator(this); };
    Iterator end() { return Iterator(nullptr); };
};

using RVIterator = JavaIteratorObject::Iterator;

template <typename Range>
class ScalaStagingIterator : public NativeObj {
  private:
    const Range range_;
    const typename Range::Iterator it_ = range_.begin();
    const typename Range::Iterator end_ = range_.end();
  public:
    template <typename ... Args>
    ScalaStagingIterator(Args ... args) : range_(args ...) { }
    char const * get() { return *it_; }
    bool advance() { return ++it_ != end_; }
};

}

#endif