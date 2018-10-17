#include <jni.h>
#include "hail/Encoder.h"
#include "hail/Upcalls.h"

namespace hail {
OutputStream::OutputStream(UpcallEnv up, jobject joutput_stream) :
  up_(up),
  joutput_stream_(joutput_stream),
  jbuf_(nullptr),
  jbuf_size_(-1) {

}

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
}