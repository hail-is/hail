#include "hail/Encoder.h"
#include "hail/Upcalls.h"
#include <jni.h>
#include <cstring>
#include <cstdio>
#include "lz4.h"

namespace hail {
//OutputStream
OutputStream::OutputStream(UpcallEnv up, jobject joutput_stream) :
  up_(up),
  joutput_stream_(up.env()->NewGlobalRef(joutput_stream)),
  jbuf_(nullptr),
  jbuf_size_(-1) { }

void OutputStream::write(char * buf, int n) {
  if (jbuf_size_ < n) {
    if (jbuf_ != nullptr) {
      up_.env()->DeleteGlobalRef(jbuf_);
    }
    auto jbuf = up_.env()->NewByteArray(n);
    jbuf_ = up_.env()->NewGlobalRef(jbuf);
    jbuf_size_ = n;
  }
  auto byteBuf = up_.env()->GetByteArrayElements(reinterpret_cast<jbyteArray>(jbuf_), nullptr);
  memcpy(byteBuf, buf, n);
  up_.env()->ReleaseByteArrayElements(reinterpret_cast<jbyteArray>(jbuf_), byteBuf, 0);
  up_.env()->CallVoidMethod(joutput_stream_, up_.config()->OutputStream_write_, jbuf_, 0, n);
}

void OutputStream::flush() {
  up_.env()->CallVoidMethod(joutput_stream_, up_.config()->OutputStream_flush_);
}

void OutputStream::close() {
  up_.env()->CallVoidMethod(joutput_stream_, up_.config()->OutputStream_close_);
}

OutputStream::~OutputStream() {
  if (jbuf_ != nullptr) {
    up_.env()->DeleteGlobalRef(jbuf_);
  }
  up_.env()->DeleteGlobalRef(joutput_stream_);
}

// StreamOutputBlockBuffer
StreamOutputBlockBuffer::StreamOutputBlockBuffer(std::shared_ptr<OutputStream> os) :
  output_stream_(os) { }

void StreamOutputBlockBuffer::write_block(char * buf, int n) {
  output_stream_->write(reinterpret_cast<char *>(&n), 4);
  output_stream_->write(buf, n);
}
}