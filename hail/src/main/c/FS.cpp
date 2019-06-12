#include "hail/FS.h"
#include "hail/Upcalls.h"
#include <jni.h>

namespace hail {

FS::FS(jobject jfs) : jfs_(up_.env()->NewGlobalRef(jfs)) { }

FS::~FS() { up_.env()->DeleteGlobalRef(jfs_); }

std::shared_ptr<OutputStream> FS::unsafe_writer(std::string path) {
  jstring jpath = up_.env()->NewStringUTF(path.c_str());
  jobject joutput_stream = up_.env()->CallObjectMethod(jfs_, unsafe_writer_method_id_, jpath);
  up_.env()->DeleteLocalRef(jpath);

  return std::make_shared<OutputStream>(up_, joutput_stream);
}

} // namespace hail