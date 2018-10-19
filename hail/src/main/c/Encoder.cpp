#include "hail/Encoder.h"
#include "hail/Upcalls.h"
#include <jni.h>
#include "lz4.h"

namespace hail {
OutputStream::OutputStream(UpcallEnv up, jobject joutput_stream) :
  up_(up),
  joutput_stream_(joutput_stream),
  jbuf_(nullptr),
  jbuf_size_(-1) {

}

//OutputStream
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

// StreamOutputBlockBuffer
StreamOutputBlockBuffer::StreamOutputBlockBuffer(OutputStream os) :
  output_stream_(os) { }

void StreamOutputBlockBuffer::write_block(char * buf, int n) {
  output_stream_.write(reinterpret_cast<char *>(&n), 4);
  output_stream_.write(buf, n);
}

// LZ4OutputBlockBuffer
LZ4OutputBlockBuffer::LZ4OutputBlockBuffer(int block_size, OutputBlockBuffer * buf) :
  block_buf_(buf),
  block_size_(LZ4_compressBound(block_size)),
  block_(new char[block_size_ + 4]{}) { }

void LZ4OutputBlockBuffer::write_block(char * buf, int n) {
  int comp_length = LZ4_compress_default(buf, block_ + 4, n, block_size_ + 4);
  reinterpret_cast<int *>(block_)[0] = n;
  block_buf_->write_block(block_, comp_length + 4);
}

// BlockingOutputBuffer
BlockingOutputBuffer::BlockingOutputBuffer(int block_size, OutputBlockBuffer * buf) :
  block_size_(block_size),
  block_buf_(buf),
  block_(new char[block_size]{}) { }

void BlockingOutputBuffer::flush() {
  if (off_ > 0) {
    block_buf_->write_block(block_, off_);
    off_ = 0;
  }
}

void BlockingOutputBuffer::write_byte(char c) {
  if (off_ + 1 > block_size_) {
    flush();
  }
  block_[off_] = c;
  off_ += 1;
}

void BlockingOutputBuffer::write_int(int i) {
  if (off_ + 4 > block_size_) {
    flush();
  }
  memcpy(block_ + off_, reinterpret_cast<char *>(&i), 4);
  off_ += 4;
}

void BlockingOutputBuffer::write_long(long l) {
  if (off_ + 8 > block_size_) {
    flush();
  }
  memcpy(block_ + off_, reinterpret_cast<char *>(&l), 8);
  off_ += 8;
}

void BlockingOutputBuffer::write_float(float f) {
  if (off_ + 4 > block_size_) {
    flush();
  }
  memcpy(block_ + off_, reinterpret_cast<char *>(&f), 4);
  off_ += 4;
}

void BlockingOutputBuffer::write_double(double d) {
  if (off_ + 8 > block_size_) {
    flush();
  }
  memcpy(block_ + off_, reinterpret_cast<char *>(&d), 8);
  off_ += 8;
}

void BlockingOutputBuffer::write_bytes(char * buf, int n) {
  int n_left = n;
  while (n_left > block_size_ - off_) {
    memcpy(block_ + off_, buf + (n - n_left), block_size_ - off_);
    n_left -= block_size_ - off_;
    off_  = block_size_;
    flush();
  }
  memcpy(block_ + off_, buf + (n - n_left), n_left);
  off_ += n_left;
}

// LEB128OutputBuffer
LEB128OutputBuffer::LEB128OutputBuffer(OutputBuffer * buf) :
  buf_(buf) { }

void LEB128OutputBuffer::flush() { buf_->flush(); }

void LEB128OutputBuffer::write_byte(char c) { buf_->write_byte(c); }

void LEB128OutputBuffer::write_int(int i) {
  unsigned int unpacked = i;
  do {
    unsigned char b = unpacked & 0x7f;
    unpacked >>= 7;
    if (unpacked != 0) {
      b |= 0x80;
    }
    buf_->write_byte(static_cast<char>(b));
  } while (unpacked != 0);
}

void LEB128OutputBuffer::write_long(long l) {
  unsigned long unpacked = l;
  do {
    unsigned char b = unpacked & 0x7f;
    unpacked >>= 7;
    if (unpacked != 0) {
      b |= 0x80;
    }
    buf_->write_byte(static_cast<char>(b));
  } while (unpacked != 0);
}

void LEB128OutputBuffer::write_float(float f) { buf_->write_float(f); }

void LEB128OutputBuffer::write_double(double d) { buf_->write_double(d); }

void LEB128OutputBuffer::write_bytes(char * buf, int n) { buf_->write_bytes(buf, n); }

}