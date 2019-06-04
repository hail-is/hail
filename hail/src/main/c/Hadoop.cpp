#include "hail/Hadoop.h"
#include "hail/Upcalls.h"
#include <jni.h>

namespace hail {

HadoopConfig::HadoopConfig(jobject jfs) : jfs_(up_.env()->NewGlobalRef(jfs)) { }

HadoopConfig::~HadoopConfig() { up_.env()->DeleteGlobalRef(jfs_); }

std::shared_ptr<OutputStream> HadoopConfig::unsafe_writer(std::string path) {
  jstring jpath = up_.env()->NewStringUTF(path.c_str());
  jobject joutput_stream = up_.env()->CallObjectMethod(jfs_, unsafe_writer_method_id_, jpath);
  up_.env()->DeleteLocalRef(jpath);

  return std::make_shared<OutputStream>(up_, joutput_stream);
}

} // namespace hail