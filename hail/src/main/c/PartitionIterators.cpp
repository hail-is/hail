#include "hail/PartitionIterators.h"
#include <cstring>
#include <jni.h>

namespace hail {
JavaIteratorWrapper::JavaIteratorWrapper(JavaIteratorWrapper &it) :
up_(it.up_),
jrvit_(it.up_.env()->NewGlobalRef(it.jrvit_)),
row_(nullptr) { }

JavaIteratorWrapper::JavaIteratorWrapper(JavaIteratorWrapper &&it) :
up_(it.up_),
jrvit_(std::move(it.jrvit_)),
row_(nullptr) { }

JavaIteratorWrapper::JavaIteratorWrapper(UpcallEnv up, jobject jrvit) :
up_(up),
jrvit_(up.env()->NewGlobalRef(jrvit)),
row_(nullptr) {
  advance();
}

bool JavaIteratorWrapper::advance() {
  if (jrvit_ == nullptr) {
    return false;
  }
  bool has_next = up_.env()->CallBooleanMethod(jrvit_, up_.config()->RVIterator_hasNext_);
  if (has_next) {
    row_ = reinterpret_cast<char *>(up_.env()->CallLongMethod(jrvit_, up_.config()->RVIterator_next_));
  } else {
    row_ = nullptr;
    up_.env()->DeleteGlobalRef(jrvit_);
    jrvit_ = nullptr;
  }
  return has_next;
}

char const* JavaIteratorWrapper::get() { return row_; }

JavaIteratorWrapper::~JavaIteratorWrapper() {
  if (jrvit_ != nullptr) {
    up_.env()->DeleteGlobalRef(jrvit_);
    jrvit_ = nullptr;
  }
}

JavaRVIterator::JavaRVIterator(JavaIteratorWrapper * jrvit) :
jit_(jrvit) { }

JavaRVIterator& JavaRVIterator::operator++() {
  if (jit_ != nullptr && !jit_->advance()) {
    jit_ = nullptr;
  }
  return *this;
}

char const* JavaRVIterator::operator*() {
  return jit_->get();
}

}