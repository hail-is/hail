#ifndef HAIL_ENCODER_H
#define HAIL_ENCODER_H 1
#include <jni.h>
#include "hail/Upcalls.h"

namespace hail {

class OutputStream {
  private:
    UpcallEnv up_;
    jobject joutput_stream_;
    jbyteArray jbuf_;
    int jbuf_size_;

  public:
    OutputStream(UpcallEnv up, jobject joutput_stream);
    void write(char * buf, int n);
    void flush();
    void close();
};

class OutputBlockBuffer {
  public:
    OutputBlockBuffer() = default;
    virtual void write_block(char * buf, int n) = 0;
    virtual void close() = 0;
};

class StreamOutputBlockBuffer : public OutputBlockBuffer {
  private:
    OutputStream output_stream_;

  public:
    StreamOutputBlockBuffer(OutputStream os);
    virtual void write_block(char * buf, int n) override;
    virtual void close() override { output_stream_.close(); };
};

class LZ4OutputBlockBuffer : public OutputBlockBuffer {
  private:
    OutputBlockBuffer * block_buf_;
    int block_size_;
    char * block_;
  public:
    LZ4OutputBlockBuffer(int block_size, OutputBlockBuffer * buf);
    virtual void write_block(char * buf, int n) override;
    virtual void close() override { block_buf_->close(); };
};

class OutputBuffer {
  public:
    OutputBuffer() = default;
    virtual void flush() = 0;
    virtual void close() = 0;
    virtual void write_byte(char c) = 0;
    virtual void write_int(int i) = 0;
    virtual void write_long(long l) = 0;
    virtual void write_float(float f) = 0;
    virtual void write_double(double d) = 0;
    virtual void write_bytes(char * buf, int n) = 0;
    virtual void write_boolean(bool z) { write_byte(z ? 1 : 0); };
};

class BlockingOutputBuffer : public OutputBuffer {
  private:
    int block_size_;
    OutputBlockBuffer * block_buf_;
    char * block_;
    int off_ = 0;

  public:
    BlockingOutputBuffer(int block_size, OutputBlockBuffer * buf);
    virtual void flush() override;
    virtual void write_byte(char c) override;
    virtual void write_int(int i) override;
    virtual void write_long(long l) override;
    virtual void write_float(float f) override;
    virtual void write_double(double d) override;
    virtual void write_bytes(char * buf, int n) override;
    virtual void close() override  { flush(); block_buf_->close(); };
};

class LEB128OutputBuffer : public OutputBuffer {
  private:
    OutputBuffer * buf_;

  public:
    LEB128OutputBuffer(OutputBuffer * buf);
    virtual void flush() override;
    virtual void write_byte(char c) override;
    virtual void write_int(int i) override;
    virtual void write_long(long l) override;
    virtual void write_float(float f) override;
    virtual void write_double(double d) override;
    virtual void write_bytes(char * buf, int n) override;
    virtual void close() override  { buf_->close(); };
};

}

#endif