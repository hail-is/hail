#ifndef HAIL_HADOOP_H
#define HAIL_HADOOP_H 1

#include "hail/Upcalls.h"
#include "hail/Encoder.h"

namespace hail {

class HadoopConfig {
  private:
    UpcallEnv up_;
    jobject jhadoop_config_;

    jclass rich_hadoop_conf_class_ = up_.env()->FindClass("is/hail/utils/richUtils/RichHadoopConfiguration");
    jmethodID unsafe_writer_method_id_ = up_.env()->GetMethodID(rich_hadoop_conf_class_, "unsafeWriter",
        "(Ljava/lang/String;)Ljava/io/OutputStream;");

  public:
    HadoopConfig(UpcallEnv up, jobject jhadoop_config);
    ~HadoopConfig();
    std::shared_ptr<OutputStream> unsafe_writer(const char *path);
};

}

#endif
