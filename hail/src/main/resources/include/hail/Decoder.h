#ifndef HAIL_DECODER_H
#define HAIL_DECODER_H 1
#include "lz4.h"
#include "hail/Upcalls.h"
#include "hail/Utils.h"
#include "hail/NativeObj.h"
#include "hail/Region.h"
#include "hail/NativeStatus.h"
#include <jni.h>
#include <memory>
#include <cstring>
#include <cassert>

namespace hail {

class InputStream {
  private:
    UpcallEnv up_;
    jobject jinput_stream_;
    jobject jbuf_;
    int jbuf_size_;

  public:
    InputStream() = delete;
    InputStream(InputStream &is) = delete;
    InputStream(UpcallEnv up, jobject jinput_stream);
    int read(char * buf, int n);
    long skip(long n);
    void close();
    ~InputStream();
};

class ByteArrayInputStream {
  private:
    char * bytes;
    long cursor;
    long size;
  public:
    ByteArrayInputStream() = delete;
    ByteArrayInputStream(ByteArrayInputStream &is) = delete;
    ByteArrayInputStream(char *bytes, long size) :
      bytes{bytes}, cursor{0}, size{size} {}

    int read(char *dest, int n) {
      auto count = std::min(static_cast<long>(n), size - cursor);
      std::memcpy(dest, bytes + cursor, count);
      cursor += count;
      return count;
    }

    long skip(long n) {
      auto diff = std::min(size - cursor, n);
      cursor += diff;
      return diff;
    }

    void close() {
      return;
    }

    ~ByteArrayInputStream() = default;
};

template <typename IS>
class StreamInputBlockBuffer {
  private:
    std::shared_ptr<IS> input_stream_;
    char len_buf_[4];

  public:
    StreamInputBlockBuffer() = delete;
    StreamInputBlockBuffer(StreamInputBlockBuffer &buf) = delete;
    StreamInputBlockBuffer(std::shared_ptr<IS> is) :
      input_stream_(is) { }

    int read_block(char * buf) {
      auto r = input_stream_->read(len_buf_, 4);
      if (r == -1) {
        return -1;
      }
      int len = load_int(len_buf_);
      return input_stream_->read(buf, len);
    }
};

template <typename IS>
class StreamInputBuffer {
  private:
    std::shared_ptr<IS> is_;
    char buf_[8];

  public:
    StreamInputBuffer() = delete;
    StreamInputBuffer(StreamInputBuffer &buf) = delete;
    StreamInputBuffer(std::shared_ptr<IS> is) : is_(is) {}

    char read_byte() {
      is_->read(buf_, 1);
      return load_byte(buf_);
    }

    int read_int() {
      is_->read(buf_, 4);
      return load_int(buf_);
    }

    long read_long() {
      is_->read(buf_, 8);
      return load_long(buf_);
    }

    float read_float() {
      is_->read(buf_, 4);
      return load_float(buf_);
    }

    double read_double() {
      is_->read(buf_, 8);
      return load_double(buf_);
    }

    void read_bytes(char * to_buf, int n) {
      is_->read(to_buf, n);
    }

    void skip_byte() { is_->skip(1); }
    void skip_int() { is_->skip(4); }
    void skip_long() { is_->skip(8); }
    void skip_float() { is_->skip(4); }
    void skip_double() { is_->skip(8); }
    void skip_bytes(int n) { is_->skip(n); }
    void skip_boolean() { skip_byte(); }
    bool read_boolean() { return read_byte() != 0; };
};

template <int BUFSIZE, typename InputBlockBuffer, typename IS>
class LZ4InputBlockBuffer {
  private:
    InputBlockBuffer block_buf_;
    char block_[BUFSIZE];
  public:
    LZ4InputBlockBuffer() = delete;
    LZ4InputBlockBuffer(LZ4InputBlockBuffer &buf) = delete;
    LZ4InputBlockBuffer(std::shared_ptr<IS> is) :
      block_buf_(is) { }

    int read_block(char * buf);
};

template <int BUFSIZE, typename InputBlockBuffer, typename IS>
int LZ4InputBlockBuffer<BUFSIZE, InputBlockBuffer, IS>::read_block(char * buf) {
  int n_read = block_buf_.read_block(block_);
  if (n_read == -1) {
    return -1;
  }
  auto decomp_len = load_int(block_);
  LZ4_decompress_safe(block_ + 4, buf, n_read - 4, decomp_len);
  return decomp_len;
};

template <int BLOCKSIZE, typename InputBlockBuffer, typename IS>
class BlockingInputBuffer {
  private:
    InputBlockBuffer block_buf_;
    char block_[BLOCKSIZE];
    int off_ = 0;
    int end_ = 0;

    void read_block() {
      end_ = block_buf_.read_block(block_);
      off_ = 0;
    }

    void read_if_empty() {
      if (UNLIKELY(off_ == end_)) {
        read_block();
      }
    }

  public:
    BlockingInputBuffer() = delete;
    BlockingInputBuffer(BlockingInputBuffer &buf) = delete;
    BlockingInputBuffer(std::shared_ptr<IS> is) :
      block_buf_(is) { }

    char read_byte() {
      read_if_empty();
      char b = load_byte(block_ + off_);
      off_ += 1;
      return b;
    }

    int read_int() {
      read_if_empty();
      int i = load_int(block_ + off_);
      off_ += 4;
      return i;
    }

    long read_long() {
      read_if_empty();
      auto l = load_long(block_ + off_);
      off_ += 8;
      return l;
    }

    float read_float() {
      read_if_empty();
      auto f = load_float(block_ + off_);
      off_ += 4;
      return f;
    }

    double read_double() {
      read_if_empty();
      auto d = load_double(block_ + off_);
      off_ += 8;
      return d;
    }

    void read_bytes(char * to_buf, int n) {
      int n_left = n;
      char * pos = to_buf;
      while (n_left > 0) {
        read_if_empty();
        int n_to_read = end_ - off_ < n_left ? end_ - off_ : n_left;
        memcpy(pos, block_ + off_, n_to_read);
        pos += n_to_read;
        off_ += n_to_read;
        n_left -= n_to_read;
      }
    }

    void skip_byte() { read_if_empty(); off_ += 1; }
    void skip_int() { read_if_empty(); off_ += 4; }
    void skip_long() { read_if_empty(); off_ += 8; }
    void skip_float() { read_if_empty(); off_ += 4; }
    void skip_double() { read_if_empty(); off_ += 8; }
    void skip_bytes(int n) {
      int n_left = n;
      while (n_left > 0) {
        read_if_empty();
        int n_to_read = ((end_ - off_) < n_left) ? end_ - off_ : n_left;
        off_ += n_to_read;
        n_left -= n_to_read;
      }
    }
    void skip_boolean() { skip_byte(); }
    bool read_boolean() { return read_byte() != 0; };
};

template <typename InputBuffer, typename IS>
class LEB128InputBuffer {
  private:
    InputBuffer buf_;

  public:
    LEB128InputBuffer() = delete;
    LEB128InputBuffer(LEB128InputBuffer &buf) = delete;
    LEB128InputBuffer(std::shared_ptr<IS> is) :
      buf_(is) { }

    char read_byte() { return buf_.read_byte(); }

    int read_int() {
      char b = read_byte();
      int x = b & 0x7f;
      int shift = 7;
      while (UNLIKELY(b & 0x80)) {
        b = read_byte();
        x |= ((b & 0x7f) << shift);
        shift += 7;
      }
      return x;
    }

    long read_long() {
      char b = read_byte();
      long x = b & 0x7f;
      int shift = 7;
      while (UNLIKELY(b & 0x80)) {
        b = read_byte();
        x |= ((b & 0x7fL) << shift);
        shift += 7;
      }
      return x;
    }
    float read_float() { return buf_.read_float(); }
    double read_double() { return buf_.read_double(); }
    void read_bytes(char * to_buf, int n) { buf_.read_bytes(to_buf, n); }
    void skip_byte() { buf_.skip_byte(); }
    void skip_int() {
      char b = read_byte();
      while (UNLIKELY(b & 0x80)) {
        b = read_byte();
      }
    }
    void skip_long() { skip_int(); }
    void skip_float() { buf_.skip_float(); }
    void skip_double() { buf_.skip_double(); }
    void skip_bytes(int n) { buf_.skip_bytes(n); }
    void skip_boolean() { buf_.skip_boolean(); }
    bool read_boolean() { return buf_.read_boolean(); }
};

template<typename Decoder>
class Reader {
private:
  Decoder dec_;
  ScalaRegion * region_;
  mutable char * value_;

  bool read() const {
    if (dec_.decode_byte()) {
      value_ = dec_.decode_row(region_->get_wrapped_region());
    } else {
      value_ = nullptr;
    }
  return (value_ != nullptr);
  }

  char * get() const { return value_; }

public:
  Reader(Decoder dec, ScalaRegion * region) :
  dec_(dec), region_(region), value_(nullptr) {
    read();
  }

  class Iterator {
  friend class Reader;
  private:
    const Reader<Decoder> * reader_;
    explicit Iterator(const Reader<Decoder> * reader) :
    reader_(reader) { }
    explicit Iterator(std::nullptr_t) : reader_(nullptr) { }

  public:
    Iterator& operator++() {
      if (reader_ != nullptr && !(reader_->read())) {
        reader_ = nullptr;
      }
      return *this;
    }

    char const* operator*() const { return reader_->get(); }

    friend bool operator==(Iterator const& lhs, Iterator const& rhs) {
      return (lhs.reader_ == rhs.reader_);
    }

    friend bool operator!=(Iterator const& lhs, Iterator const& rhs) {
      return !(lhs == rhs);
    }
  };

  Iterator begin() const { return Iterator(this); }
  Iterator end() const { return Iterator(nullptr); }
};

}

#endif
