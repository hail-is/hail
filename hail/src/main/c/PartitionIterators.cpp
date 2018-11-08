#include "hail/PartitionIterators.h"
#include <cstring>
#include <jni.h>

namespace hail {
JavaIteratorWrapper::JavaIteratorWrapper(JavaIteratorWrapper &it) :
up_(it.up_),
jrvit_(it.up_.env()->NewGlobalRef(it.jrvit_)),
n_read_(it.n_read_) { }

JavaIteratorWrapper::JavaIteratorWrapper(JavaIteratorWrapper &&it) :
up_(it.up_),
jrvit_(std::move(it.jrvit_)),
n_read_(it.n_read_) { }

JavaIteratorWrapper::JavaIteratorWrapper(UpcallEnv up, jobject jrvit) :
up_(up),
jrvit_(up.env()->NewGlobalRef(jrvit)),
n_read_(0) { }

char * JavaIteratorWrapper::next() {
  return reinterpret_cast<char *>(up_.env()->CallLongMethod(jrvit_, up_.config()->RVIterator_next_));
}

bool JavaIteratorWrapper::has_next() {
  return up_.env()->CallBooleanMethod(jrvit_, up_.config()->RVIterator_hasNext_);
}

JavaIteratorWrapper::~JavaIteratorWrapper() {
  up_.env()->DeleteGlobalRef(jrvit_);
}

JavaRVIterator::JavaRVIterator(std::shared_ptr<JavaIteratorWrapper> jrvit) :
jit_(jrvit),
row_(nullptr) { }

JavaRVIterator& JavaRVIterator::operator++() {
  if (jit_ != nullptr) {
    if (jit_->has_next()) {
        row_ = jit_->next();
    } else {
      row_ = nullptr;
      jit_ = nullptr;
    }
  }
  return *this;
}

bool JavaRVIterator::operator==(JavaRVIterator other) const {
  return ((jit_ == nullptr) && (other.jit_ == nullptr)) || ((jit_ == other.jit_) && (row_ == other.row_));
}

bool JavaRVIterator::operator!=(JavaRVIterator other) const {
  return !(*this == other);
}

char * JavaRVIterator::operator*() const {
  return row_;
}

JavaRVIterator JavaRVIterator::begin() const {
  return JavaRVIterator(jit_);
}

JavaRVIterator JavaRVIterator::end() const {
  return JavaRVIterator(nullptr);
}

}