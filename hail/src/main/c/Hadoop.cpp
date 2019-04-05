#include "hail/Hadoop.h"

namespace hail {

HadoopConfig::HadoopConfig(UpcallEnv up, jobject jhadoop_config) :
  up_(up), jhadoop_config_(jhadoop_config) { }

HadoopConfig::unsafe_writer(const char *path) {
  auto jpath = up_.env()->NewStringUTF(path);
  auto joutput_stream = up_.env()->CallObjectMethod(jhadoop_config_, up_.config()->RichHadoopConfiguration_unsafeWriter_, jpath);

  return OutputStream(up_, joutput_stream);
}

}