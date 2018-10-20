#ifndef HAIL_ENCODER_H
#define HAIL_ENCODER_H 1
#include <jni.h>
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

class OutputBlockBuffer {
  public:
    OutputBlockBuffer() = default;
    OutputBlockBuffer(OutputBlockBuffer * src) = delete;
    virtual void clone(const OutputBlockBuffer * src) = 0;
    virtual void write_block(char * buf, int n) = 0;
    virtual void close() = 0;
};

class StreamOutputBlockBuffer : public OutputBlockBuffer {
  private:
    std::shared_ptr<OutputStream> output_stream_;

  public:
    StreamOutputBlockBuffer(OutputStream os);
    StreamOutputBlockBuffer(StreamOutputBlockBuffer * src);
    virtual void clone(const OutputBlockBuffer * src) override;
    virtual void write_block(char * buf, int n) override;
    virtual void close() override { output_stream_->close(); };
};

class LZ4OutputBlockBuffer : public OutputBlockBuffer {
  private:
    std::shared_ptr<OutputBlockBuffer> block_buf_;
    int block_size_;
    char * block_;
  public:
    LZ4OutputBlockBuffer(int block_size, std::shared_ptr<OutputBlockBuffer> buf);
    LZ4OutputBlockBuffer(LZ4OutputBlockBuffer * src);
    virtual void clone(const OutputBlockBuffer * src) override;
    virtual void write_block(char * buf, int n) override;
    virtual void close() override { block_buf_->close(); };
};

class OutputBuffer : public NativeObj {
  public:
    OutputBuffer() = default;
    OutputBuffer(const OutputBuffer * src) = delete;
    virtual void clone(const OutputBuffer * src) = 0;
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
    std::shared_ptr<OutputBlockBuffer> block_buf_;
    char * block_;
    int off_ = 0;

  public:
    BlockingOutputBuffer(BlockingOutputBuffer * src);
    BlockingOutputBuffer(int block_size, std::shared_ptr<OutputBlockBuffer> buf);
    virtual void clone(const OutputBuffer * src) override;
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
    std::shared_ptr<OutputBuffer> buf_;

  public:
    LEB128OutputBuffer(std::shared_ptr<OutputBuffer> buf);
    LEB128OutputBuffer(LEB128OutputBuffer * buf);
    virtual void clone(const OutputBuffer * src) override;
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