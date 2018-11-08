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
    long n_read_;

  public:
    JavaIteratorWrapper() = delete;
    JavaIteratorWrapper(JavaIteratorWrapper &it);
    JavaIteratorWrapper(JavaIteratorWrapper &&it);
    JavaIteratorWrapper(UpcallEnv up, jobject jrvit);
    char * next();
    bool has_next();
    ~JavaIteratorWrapper();

};

class JavaRVIterator {
  private:
    std::shared_ptr<JavaIteratorWrapper> jit_;
    char * row_;
    long n_read_;

  public:
    JavaRVIterator() = delete;
    JavaRVIterator(std::shared_ptr<JavaIteratorWrapper> jrvit);
    JavaRVIterator& operator++();
    bool operator==(JavaRVIterator other) const;
    bool operator!=(JavaRVIterator other) const;
    char * operator*() const;
    JavaRVIterator begin() const;
    JavaRVIterator end() const;
};

}

#endif