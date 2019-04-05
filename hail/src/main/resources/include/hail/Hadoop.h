#ifndef HAIL_HADOOP_H
#define HAIL_HADOOP_H 1

#include "hail/Upcalls.h"
#include "hail/Encoder.h"

namespace hail {

class HadoopConfig {
  private:
    UpcallEnv up_;
    jobject jhadoop_config_;

  public:
    HadoopConfig(UpcallEnv up, jobject jhadoop_config);
    OutputStream unsafe_writer(const char *path);
};

HadoopConfig::HadoopConfig(UpcallEnv up, jobject jhadoop_config) :
  up_(up), jhadoop_config_(jhadoop_config) { }

OutputStream HadoopConfig::unsafe_writer(const char *path) {
  jstring jpath = up_.env()->NewStringUTF(path);
  jobject joutput_stream = up_.env()->CallObjectMethod(jhadoop_config_, up_.config()->RichHadoopConfiguration_unsafeWriter_, jpath);

  return { up_, joutput_stream };
}

}

#endif

