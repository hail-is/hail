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

int InputStream::read(char * buf, int n) {
  if (UNLIKELY(n == 0)) {
    return 0;
  }

  if (jbuf_size_ < n) {
    if (jbuf_ != nullptr) {
      up_.env()->DeleteGlobalRef(jbuf_);
    }
    auto jbuf = up_.env()->NewByteArray(std::max(n, 1024));
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

ByteArrayInputStream::ByteArrayInputStream(char *bytes, long size) :
  bytes{bytes}, cursor{0}, size{size} {}

int ByteArrayInputStream::read(char *dest, int n) {
  auto count = std::min(static_cast<long>(n), size - cursor);
  std::memcpy(dest, bytes, count);
  return count;
}

long ByteArrayInputStream::skip(long n) {
  auto diff = std::min(size - cursor, n);
  cursor += diff;
  return diff;
}

void ByteArrayInputStream::close() {
  return;
}

}
