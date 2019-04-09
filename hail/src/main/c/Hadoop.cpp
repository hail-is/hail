#include "hail/Hadoop.h"
#include "hail/Upcalls.h"
#include <jni.h>

namespace hail {

HadoopConfig::HadoopConfig(UpcallEnv up, jobject jhadoop_config) :
  up_(up),
  jhadoop_config_(up.env()->NewGlobalRef(jhadoop_config)) { }

HadoopConfig::~HadoopConfig() { up_.env()->DeleteGlobalRef(jhadoop_config_); }

std::shared_ptr<OutputStream> HadoopConfig::unsafe_writer(const char *path) {
  jstring jpath = up_.env()->NewStringUTF(path);
  jobject joutput_stream = up_.env()->CallObjectMethod(jhadoop_config_, unsafe_writer_method_id_, jpath);
  up_.env()->DeleteLocalRef(jpath);

  return std::make_shared<OutputStream>(up_, joutput_stream);
}

} // namespace hail