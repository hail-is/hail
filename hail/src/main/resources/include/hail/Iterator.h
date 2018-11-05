#ifndef HAIL_ITERATOR_H
#define HAIL_ITERATOR_H 1
#include "hail/Upcalls.h"
#include "hail/Utils.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <memory>
#include <cstring>

namespace hail {

class RVIterator : public NativeObj {
  private:
    UpcallEnv up_;
    jobject jrvit_;

  public:
    RVIterator() = delete;
    RVIterator(RVIterator &it) = delete;
    RVIterator(UpcallEnv up, jobject jrvit) :
      up_(up),
      jrvit_(up.env()->NewGlobalRef(jrvit)) { }

    char * next() {
      return reinterpret_cast<char *>(up_.env()->CallLongMethod(jrvit_, up_.config()->RVIterator_next_));
    }

    bool has_next() {
      return up_.env()->CallBooleanMethod(jrvit_, up_.config()->RVIterator_hasNext_);
    }

    ~RVIterator() {
      up_.env()->DeleteGlobalRef(jrvit_);
    }
};

}

#endif