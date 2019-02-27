#ifndef HAIL_SPARKUTILS_H
#define HAIL_SPARKUTILS_H 1

#include <jni.h>
#include <cstring>
#include "hail/Decoder.h"
#include "hail/Encoder.h"
#include "hail/Region.h"
#include "hail/Upcalls.h"
#include "hail/Utils.h"

namespace hail {

class SparkEnv {
  private:
    UpcallEnv up_;
    UpcallEnv * up() { return &up_; }

  public:
    JNIEnv * env() { return up_.env(); }
};

struct SparkFunctionContext {
  RegionPtr region_;
  SparkEnv spark_env_;

  SparkFunctionContext(RegionPtr region) :
  region_(region), spark_env_() { }
};

}

#endif