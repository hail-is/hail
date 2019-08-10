#ifndef HAIL_ENCODER_H
#define HAIL_ENCODER_H 1
#include "lz4.h"
#include "hail/Upcalls.h"
#include "hail/Utils.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <memory>
#include <cstring>

namespace hail {

class OutputStream {
  private:
    UpcallEnv up_;
    jobject joutput_stream_;
    jobject jbuf_;
    int jbuf_size_;

  public:
    OutputStream() = delete;
    OutputStream(OutputStream &os) = delete;
    OutputStream(UpcallEnv up, jobject joutput_stream);
    void write(char const* buf, int n);
    void flush();
    void close();
    ~OutputStream();
};

class StreamOutputBlockBuffer {
  private:
    std::shared_ptr<OutputStream> output_stream_;

  public:
    StreamOutputBlockBuffer() = delete;
    StreamOutputBlockBuffer(StreamOutputBlockBuffer &buf) = delete;
    StreamOutputBlockBuffer(std::shared_ptr<OutputStream> os);
    void write_block(char const* buf, int n);
    void close() { output_stream_->close(); }
};

class StreamOutputBuffer {
  private:
    std::shared_ptr<OutputStream> os_;

  public:
    StreamOutputBuffer() = delete;
    StreamOutputBuffer(StreamOutputBuffer &buf) = delete;
    StreamOutputBuffer(std::shared_ptr<OutputStream> os) : os_(os) {}

    void flush() {
      os_->flush();
    }

    void write_byte(char c) {
      os_->write(reinterpret_cast<char const*>(&c), 1);
    }

    void write_boolean(bool b) { write_byte(b ? 1 : 0); };

    void write_int(int i) {
      os_->write(reinterpret_cast<char const*>(&i), 4);
    }

    void write_long(long l) {
      os_->write(reinterpret_cast<char const*>(&l), 8);
    }

    void write_float(float f) {
      os_->write(reinterpret_cast<char const*>(&f), 4);
    }

    void write_double(double d) {
      os_->write(reinterpret_cast<char const*>(&d), 8);
    }

    void write_bytes(char const* buf, int n) {
      os_->write(buf, n);
    }

    void close() { flush(); os_->close(); }
};

template <int BUFSIZE, typename OutputBlockBuffer>
class LZ4OutputBlockBuffer {
  private:
    OutputBlockBuffer block_buf_;
    char block_[BUFSIZE];
  public:
    LZ4OutputBlockBuffer() = delete;
    LZ4OutputBlockBuffer(LZ4OutputBlockBuffer &buf) = delete;
    LZ4OutputBlockBuffer(std::shared_ptr<OutputStream> out) :
      block_buf_(out) { }

    void write_block(char const* buf, int n) {
      int comp_length = LZ4_compress_default(buf, block_ + 4, n, BUFSIZE);
      store_int(block_, n);
      block_buf_.write_block(block_, comp_length + 4);
    }

    void close() { block_buf_.close(); }
};

template <int BLOCKSIZE, typename OutputBlockBuffer>
class BlockingOutputBuffer {
  private:
    OutputBlockBuffer block_buf_;
    char block_[BLOCKSIZE];
    int off_ = 0;

  public:
    BlockingOutputBuffer() = delete;
    BlockingOutputBuffer(BlockingOutputBuffer &buf) = delete;
    BlockingOutputBuffer(std::shared_ptr<OutputStream> out) :
      block_buf_(out) { }

    void flush() {
      if (off_ > 0) {
        block_buf_.write_block(block_, off_);
        off_ = 0;
      }
    }

    void write_byte(char c) {
      if (off_ + 1 > BLOCKSIZE) {
        flush();
      }
      block_[off_] = c;
      off_ += 1;
    }

    void write_boolean(bool b) { write_byte(b ? 1 : 0); };

    void write_int(int i) {
      if (off_ + 4 > BLOCKSIZE) {
        flush();
      }
      store_int(block_ + off_, i);
      off_ += 4;
    }

    void write_long(long l) {
      if (off_ + 8 > BLOCKSIZE) {
        flush();
      }
      store_long(block_ + off_, l);
      off_ += 8;
    }

    void write_float(float f) {
      if (off_ + 4 > BLOCKSIZE) {
        flush();
      }
      store_float(block_ + off_, f);
      off_ += 4;
    }

    void write_double(double d) {
      if (off_ + 8 > BLOCKSIZE) {
        flush();
      }
      store_double(block_ + off_, d);
      off_ += 8;
    }

    void write_bytes(char const* buf, int n) {
      int n_left = n;
      while (n_left > BLOCKSIZE - off_) {
        memcpy(block_ + off_, buf + (n - n_left), BLOCKSIZE - off_);
        n_left -= BLOCKSIZE - off_;
        off_  = BLOCKSIZE;
        flush();
      }
      memcpy(block_ + off_, buf + (n - n_left), n_left);
      off_ += n_left;
    }

    void close() { flush(); block_buf_.close(); }
};

template <typename OutputBuffer>
class LEB128OutputBuffer {
  private:
    OutputBuffer buf_;

  public:
    LEB128OutputBuffer() = delete;
    LEB128OutputBuffer(LEB128OutputBuffer &buf) = delete;
    LEB128OutputBuffer(std::shared_ptr<OutputStream> out) :
      buf_(out) { }

    void flush() { buf_.flush(); }

    void write_byte(char c) { buf_.write_byte(c); }

    void write_boolean(bool b) { write_byte(b ? 1 : 0); }

    void write_int(int i) {
      unsigned int unpacked = i;
      do {
        unsigned char b = unpacked & 0x7f;
        unpacked >>= 7;
        if (unpacked != 0) {
          b |= 0x80;
        }
        buf_.write_byte(static_cast<char>(b));
      } while (UNLIKELY(unpacked != 0));
    }

    void write_long(long l) {
      unsigned long unpacked = l;
      do {
        unsigned char b = unpacked & 0x7f;
        unpacked >>= 7;
        if (unpacked != 0) {
          b |= 0x80;
        }
        buf_.write_byte(static_cast<char>(b));
      } while (UNLIKELY(unpacked != 0));
    }

    void write_float(float f) { buf_.write_float(f); }

    void write_double(double d) { buf_.write_double(d); }

    void write_bytes(char const* buf, int n) { buf_.write_bytes(buf, n); }

    void close() { buf_.close(); }
};

}

#endif