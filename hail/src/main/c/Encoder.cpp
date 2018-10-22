#include "hail/Encoder.h"
#include "hail/Upcalls.h"
#include <jni.h>
#include "lz4.h"

namespace hail {
//OutputStream
OutputStream::OutputStream(UpcallEnv up, jobject joutput_stream) :
  up_(up),
  joutput_stream_(up.env()->NewGlobalRef(joutput_stream)),
  jbuf_(nullptr),
  jbuf_size_(-1) { }

OutputStream::OutputStream(OutputStream * output_stream) :
  up_(output_stream->up_),
  joutput_stream_(output_stream->up_.env()->NewGlobalRef(output_stream->joutput_stream_)),
  jbuf_(nullptr),
  jbuf_size_(-1) { }

void OutputStream::write(char * buf, int n) {
  if (jbuf_size_ < n) {
    jbuf_ = up_.env()->NewByteArray(n);
    jbuf_size_ = n;
  }
  auto byteBuf = up_.env()->GetByteArrayElements(jbuf_, nullptr);
  memcpy(byteBuf, buf, n);
  up_.env()->ReleaseByteArrayElements(jbuf_, byteBuf, 0);
  up_.env()->CallVoidMethod(joutput_stream_, up_.config()->OutputStream_write_, jbuf_, 0, n);
}

void OutputStream::flush() {
  up_.env()->CallVoidMethod(joutput_stream_, up_.config()->OutputStream_flush_);
}

void OutputStream::close() {
  up_.env()->CallVoidMethod(joutput_stream_, up_.config()->OutputStream_close_);
}

OutputStream::~OutputStream() {
  up_.env()->DeleteGlobalRef(joutput_stream_);
}

// StreamOutputBlockBuffer
StreamOutputBlockBuffer::StreamOutputBlockBuffer(OutputStream os) :
  output_stream_(std::make_shared<OutputStream>(&os)) { }

StreamOutputBlockBuffer::StreamOutputBlockBuffer(StreamOutputBlockBuffer * src) :
  output_stream_(src->output_stream_) { }

void StreamOutputBlockBuffer::write_block(char * buf, int n) {
  output_stream_->write(reinterpret_cast<char *>(&n), 4);
  output_stream_->write(buf, n);
}
}