#include "hail/Hadoop.h"
#include "hail/Upcalls.h"
#include <jni.h>

namespace hail {

HadoopConfig::HadoopConfig(jobject jhadoop_config) : jhadoop_config_(up_.env()->NewGlobalRef(jhadoop_config)) { }

HadoopConfig::~HadoopConfig() { up_.env()->DeleteGlobalRef(jhadoop_config_); }

std::shared_ptr<OutputStream> HadoopConfig::unsafe_writer(std::string path) {
  jstring jpath = up_.env()->NewStringUTF(path.c_str());
  jobject joutput_stream = up_.env()->CallObjectMethod(jhadoop_config_, unsafe_writer_method_id_, jpath);
  up_.env()->DeleteLocalRef(jpath);

  return std::make_shared<OutputStream>(up_, joutput_stream);
}

} // namespace hail