#ifndef HAIL_ENCODER_H
#define HAIL_ENCODER_H 1
#include "lz4.h"
#include "hail/Upcalls.h"
#include "hail/Utils.h"
#include "hail/NativeObj.h"
#include <jni.h>
#include <memory>
#include <cstring>

#define LIKELY(condition)   __builtin_expect(static_cast<bool>(condition), 1)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

namespace hail {

class OutputStream {
  private:
    UpcallEnv up_;
    jobject joutput_stream_;
    jobject jbuf_;
    int jbuf_size_;

  public:
    OutputStream(UpcallEnv up, jobject joutput_stream);
    void write(char * buf, int n);
    void flush();
    void close();
    ~OutputStream();
};

class StreamOutputBlockBuffer {
  private:
    std::shared_ptr<OutputStream> output_stream_;

  public:
    StreamOutputBlockBuffer(std::shared_ptr<OutputStream> os);
    void write_block(char * buf, int n);
    void close() { output_stream_->close(); }
};

template <int BLOCKSIZE, typename OutputBlockBuffer>
class LZ4OutputBlockBuffer {
  private:
    OutputBlockBuffer block_buf_;
    char * block_;
  public:
    LZ4OutputBlockBuffer(std::shared_ptr<OutputStream> out) :
      block_buf_(OutputBlockBuffer(out)),
      block_(new char[LZ4_compressBound(BLOCKSIZE) + 4]{}) { }

    ~LZ4OutputBlockBuffer() { delete[] block_; }

    void write_block(char * buf, int n) {
      int comp_length = LZ4_compress_default(buf, block_ + 4, n, LZ4_compressBound(BLOCKSIZE) + 4);
      store_int(block_, n);
      block_buf_.write_block(block_, comp_length + 4);
    }

    void close() { block_buf_.close(); }
};

template <int BLOCKSIZE, typename OutputBlockBuffer>
class BlockingOutputBuffer {
  private:
    OutputBlockBuffer block_buf_;
    char * block_;
    int off_ = 0;

  public:
    BlockingOutputBuffer(std::shared_ptr<OutputStream> out) :
      block_buf_(OutputBlockBuffer(out)),
      block_(new char[BLOCKSIZE]{}) { }

    ~BlockingOutputBuffer() { delete[] block_; }

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

    void write_bytes(char * buf, int n) {
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
    LEB128OutputBuffer(std::shared_ptr<OutputStream> out) :
      buf_(OutputBuffer(out)) { }

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

    void write_bytes(char * buf, int n) { buf_.write_bytes(buf, n); }

    void close() { buf_.close(); }
};

template<typename OutputBuffer>
class Encoder : public NativeObj {
  private:
    OutputBuffer buf_;

  public:
    Encoder(std::shared_ptr<OutputStream> out) :
      buf_(OutputBuffer(out)) { }
    OutputBuffer get_buf() { return buf_; }
};

}

#endif