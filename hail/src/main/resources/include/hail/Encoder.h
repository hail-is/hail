#ifndef HAIL_ENCODER_H
#define HAIL_ENCODER_H 1
#include <jni.h>
#include "lz4.h"
#include "hail/Upcalls.h"
#include "hail/NativeObj.h"

namespace hail {

class OutputStream {
  private:
    UpcallEnv up_;
    jobject joutput_stream_;
    jbyteArray jbuf_;
    int jbuf_size_;

  public:
    OutputStream(UpcallEnv up, jobject joutput_stream);
    OutputStream(OutputStream * output_stream);
    void write(char * buf, int n);
    void flush();
    void close();
    ~OutputStream();
};

class StreamOutputBlockBuffer : public NativeObj {
  private:
    std::shared_ptr<OutputStream> output_stream_;

  public:
    StreamOutputBlockBuffer(OutputStream os);
    StreamOutputBlockBuffer(StreamOutputBlockBuffer * src);
    void write_block(char * buf, int n);
    void close() { output_stream_->close(); };
};

template <typename T>
class LZ4OutputBlockBuffer : public NativeObj {
  private:
    std::shared_ptr<T> block_buf_;
    int block_size_;
    char * block_;
  public:
    LZ4OutputBlockBuffer(int block_size, std::shared_ptr<T> buf) :
      block_buf_(buf),
      block_size_(LZ4_compressBound(block_size)),
      block_(new char[block_size_ + 4]{}) { };

    LZ4OutputBlockBuffer(LZ4OutputBlockBuffer<T> * src) :
      block_buf_(src->block_buf_),
      block_size_(src->block_size_),
      block_(new char[src->block_size_ + 4]{}) { };

    void write_block(char * buf, int n) {
      int comp_length = LZ4_compress_default(buf, block_ + 4, n, block_size_ + 4);
      reinterpret_cast<int *>(block_)[0] = n;
      block_buf_->write_block(block_, comp_length + 4);
    };

    void close() { block_buf_->close(); };
};

template <typename T>
class BlockingOutputBuffer : public NativeObj {
  private:
    int block_size_;
    std::shared_ptr<T> block_buf_;
    char * block_;
    int off_ = 0;

  public:
    BlockingOutputBuffer(int block_size, std::shared_ptr<T> buf) :
      block_size_(block_size),
      block_buf_(buf),
      block_(new char[block_size]{}) { }

    BlockingOutputBuffer(BlockingOutputBuffer<T> * src) :
      block_size_(src->block_size_),
      block_buf_(src->block_buf_),
      block_(new char[src->block_size_]{}) { }

    void flush() {
      if (off_ > 0) {
        block_buf_->write_block(block_, off_);
        off_ = 0;
      }
    }

    void write_byte(char c) {
      if (off_ + 1 > block_size_) {
        flush();
      }
      block_[off_] = c;
      off_ += 1;
    }

    void write_int(int i) {
      if (off_ + 4 > block_size_) {
        flush();
      }
      memcpy(block_ + off_, reinterpret_cast<char *>(&i), 4);
      off_ += 4;
    }

    void write_long(long l) {
      if (off_ + 8 > block_size_) {
        flush();
      }
      memcpy(block_ + off_, reinterpret_cast<char *>(&l), 8);
      off_ += 8;
    }

    void write_float(float f) {
      if (off_ + 4 > block_size_) {
        flush();
      }
      memcpy(block_ + off_, reinterpret_cast<char *>(&f), 4);
      off_ += 4;
    }

    void write_double(double d) {
      if (off_ + 8 > block_size_) {
        flush();
      }
      memcpy(block_ + off_, reinterpret_cast<char *>(&d), 8);
      off_ += 8;
    }

    void write_bytes(char * buf, int n) {
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

    void close() { flush(); block_buf_->close(); };
};

template <typename T>
class LEB128OutputBuffer : public NativeObj {
  private:
    std::shared_ptr<T> buf_;

  public:
    LEB128OutputBuffer(std::shared_ptr<T> buf) :
      buf_(buf) { }

    LEB128OutputBuffer(LEB128OutputBuffer<T> * src) :
      buf_(src->buf_) { }

    void flush() { buf_->flush(); }

    void write_byte(char c) { buf_->write_byte(c); }

    void write_int(int i) {
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

    void write_long(long l) {
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

    void write_float(float f) { buf_->write_float(f); }

    void write_double(double d) { buf_->write_double(d); }

    void write_bytes(char * buf, int n) { buf_->write_bytes(buf, n); }

    void close() { buf_->close(); };
};

}

#endif