#include <jni.h>
#include <cstring>
#include "hail/Decoder.h"
#include "lz4.h"
#include "hail/Utils.h"

namespace hail {

// InputStream
InputStream::InputStream(UpcallEnv up, jobject jinput_stream) :
  up_(up),
  jinput_stream_(up.env()->NewGlobalRef(jinput_stream)),
  jbuf_(nullptr),
  jbuf_size_(-1) { }

InputStream::InputStream(UpcallEnv up, jobject jhadoop_conf, char * path) :
  InputStream(up, up.get_input_stream(jhadoop_conf, path)) { }

int InputStream::read(char * buf, int n) {
  if (jbuf_size_ < n) {
    if (jbuf_ != nullptr) {
      up_.env()->DeleteGlobalRef(jbuf_);
    }
    auto jbuf = up_.env()->NewByteArray(n);
    jbuf_ = up_.env()->NewGlobalRef(jbuf);
    jbuf_size_ = n;
  }

  int n_read = up_.env()->CallIntMethod(jinput_stream_, up_.config()->InputStream_read_, jbuf_, 0, n);
  up_.env()->GetByteArrayRegion(reinterpret_cast<jbyteArray>(jbuf_), 0, n_read, reinterpret_cast<jbyte *>(buf));
  return n_read;
}

long InputStream::skip(long n) {
  return up_.env()->CallLongMethod(jinput_stream_, up_.config()->InputStream_skip_, n);
}

void InputStream::close() {
  up_.env()->CallVoidMethod(jinput_stream_, up_.config()->InputStream_close_);
}

InputStream::~InputStream() {
  if (jbuf_ != nullptr) {
    up_.env()->DeleteGlobalRef(jbuf_);
    jbuf_ = nullptr;
    jbuf_size_ = -1;
  }
  up_.env()->DeleteGlobalRef(jinput_stream_);
}

// StreamInputBlockBuffer
StreamInputBlockBuffer::StreamInputBlockBuffer(std::shared_ptr<InputStream> is) :
  input_stream_(is) { }

int StreamInputBlockBuffer::read_block(char * buf) {
  auto r = input_stream_->read(len_buf_, 4);
  if (r == -1) {
    return -1;
  }
  int len = load_int(len_buf_);
  return input_stream_->read(buf, len);
}

}
