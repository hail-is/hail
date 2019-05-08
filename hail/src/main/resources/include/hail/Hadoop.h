#ifndef HAIL_FS_H
#define HAIL_FS_H 1

#include "hail/Upcalls.h"
#include "hail/Encoder.h"

namespace hail {

class FS {
  private:
    UpcallEnv up_;
    jobject jfs_;

    jclass fs_class_ = up_.env()->FindClass("is/hail/io/fs/FS");
    jmethodID unsafe_writer_method_id_ = up_.env()->GetMethodID(fs_class_, "unsafeWriter",
        "(Ljava/lang/String;)Ljava/io/OutputStream;");

  public:
    FS(jobject jfs);
    ~FS();
    std::shared_ptr<OutputStream> unsafe_writer(std::string path);
};

}

#endif
