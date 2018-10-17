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

}

#endif