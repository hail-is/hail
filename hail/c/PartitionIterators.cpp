#include "hail/PartitionIterators.h"
#include <cstring>
#include <jni.h>

namespace hail {
JavaIteratorObject::JavaIteratorObject(JavaIteratorObject &it) :
up_(it.up_),
jrvit_(it.up_.env()->NewGlobalRef(it.jrvit_)),
row_(nullptr) { }

JavaIteratorObject::JavaIteratorObject(JavaIteratorObject &&it) :
up_(it.up_),
jrvit_(std::move(it.jrvit_)),
row_(nullptr) { }

JavaIteratorObject::JavaIteratorObject(UpcallEnv up, jobject jrvit) :
up_(up),
jrvit_(up.env()->NewGlobalRef(jrvit)),
row_(nullptr) {
  advance();
}

bool JavaIteratorObject::advance() {
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

char const* JavaIteratorObject::get() const { return row_; }

JavaIteratorObject::~JavaIteratorObject() {
  if (jrvit_ != nullptr) {
    up_.env()->DeleteGlobalRef(jrvit_);
    jrvit_ = nullptr;
  }
}

RVIterator::Iterator(JavaIteratorObject * jrvit) :
jit_(jrvit) { }

RVIterator& RVIterator::operator++() {
  if (jit_ != nullptr && !jit_->advance()) {
    jit_ = nullptr;
  }
  return *this;
}

char const* RVIterator::operator*() const {
  return jit_->get();
}

}