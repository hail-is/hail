#ifndef HAIL_SPARKUTILS_H
#define HAIL_SPARKUTILS_H 1

#include <jni.h>
#include <cstring>
#include "hail/Decoder.h"
#include "hail/Encoder.h"
#include "hail/Region.h"
#include "hail/Upcalls.h"
#include "hail/Utils.h"

namespace hail {

class SparkEnv {
  private:
    UpcallEnv up_;

  public:
    JNIEnv * env() { return up_.env(); }

  private:
    jclass spark_utils_class_ = env()->FindClass("is/hail/cxx/SparkUtils$");
    jfieldID spark_utils_field_ = env()->GetStaticFieldID(spark_utils_class_, "MODULE$", "Lis/hail/cxx/SparkUtils$;");
    jmethodID parallelize_compute_collect_ = up_.env()->GetMethodID(spark_utils_class_, "parallelizeComputeCollect", "(Ljava/lang/String;Ljava/lang/String;[[B[B)[[B");

    jclass bais_ = env()->FindClass("java/io/ByteArrayInputStream");
    jmethodID InputStream_constructor_ = env()->GetMethodID(bais_, "<init>", "([B)V");
    jclass baos_ = env()->FindClass("java/io/ByteArrayOutputStream");
    jmethodID OutputStream_constructor_ = env()->GetMethodID(baos_, "<init>", "()V");
    jmethodID OutputStream_toByteArray_ = env()->GetMethodID(baos_, "toByteArray", "()[B");
    jmethodID OutputStream_reset_ = env()->GetMethodID(baos_, "reset", "()V");
    jclass byteArrayClass_ = env()->FindClass("[B");

    UpcallEnv * up() { return &up_; }
    jobject spark_utils() { return env()->GetStaticObjectField(spark_utils_class_, spark_utils_field_); }


  public:
    // contexts can't be missing
    template <typename EltEncoder, typename ArrayImpl>
    class ArrayEncoder {
      private:
        SparkEnv * senv_;
        jobject baos_;
        jobject jctx_;
        EltEncoder eltEnc_;
        JNIEnv * env() { return senv_->env(); }

      public:
        ArrayEncoder(SparkEnv * senv, jobject baos) :
        senv_(senv),
        baos_(baos),
        eltEnc_(std::make_shared<OutputStream>(*(senv->up()), baos)) { }

        jobject encode(const char * ctxs) {
          auto len = ArrayImpl::load_length(ctxs);
          auto jctx = env()->NewObjectArray(len, senv_->byteArrayClass_, nullptr);
          for (int i=0; i<len; ++i) {
            if (ArrayImpl::is_element_missing(ctxs, i)) {
              throw new FatalError("context cannot be missing");
            } else {
              env()->CallVoidMethod(baos_, senv_->OutputStream_reset_);
              eltEnc_.encode_row(ArrayImpl::load_element(ctxs, i));
              eltEnc_.flush();
              env()->SetObjectArrayElement(jctx, i, env()->CallObjectMethod(baos_, senv_->OutputStream_toByteArray_));
            }
          }
          env()->CallVoidMethod(baos_, senv_->OutputStream_reset_);
          return jctx;
        }
    };

    template <typename EltDecoder, typename ArrayBuilder>
    class ArrayDecoder {
      private:
        SparkEnv * senv_;
        JNIEnv * env() { return senv_->env(); }

      public:
        ArrayDecoder(SparkEnv * senv) :
        senv_(senv) { }

        const char * decode(Region * region, jobjectArray elements) {
          auto len = env()->GetArrayLength(elements);
          ArrayBuilder builder { region, len };
          for (int i=0; i<len; ++i) {
            auto bais = env()->NewObject(senv_->bais_, senv_->InputStream_constructor_, env()->GetObjectArrayElement(elements, i));
            EltDecoder elt_dec { std::make_shared<InputStream>(*(senv_->up()), bais) };
            auto off = elt_dec.decode_row(region);
            builder.set_element(i, off);
            env()->DeleteLocalRef(bais);
          }
          return builder.offset();
        }
    };

    template <typename CtxEncoder, typename GlobalEncoder, typename ResultDecoder>
    const char * compute_distributed_array(RegionPtr region, const char * modID, const char * fname, const char * ctxs, const char * globals) {
      auto baos = env()->NewObject(baos_, OutputStream_constructor_);
      auto os = std::make_shared<OutputStream>(up_, baos);
      GlobalEncoder globalEnc { os };
      globalEnc.encode_row(globals);
      globalEnc.flush();
      auto jglobals = env()->CallObjectMethod(baos, OutputStream_toByteArray_);

      CtxEncoder ctxEnc { this, baos };
      auto jctxs = env()->NewLocalRef(ctxEnc.encode(ctxs));

      jobject jmodID = env()->NewStringUTF(modID);
      jobject jfname = env()->NewStringUTF(fname);
      auto res = env()->NewGlobalRef(env()->CallObjectMethod(spark_utils(), parallelize_compute_collect_, jmodID, jfname, jctxs, jglobals));
      env()->DeleteLocalRef(jmodID);
      env()->DeleteLocalRef(jfname);
      env()->DeleteLocalRef(jctxs);
      env()->DeleteLocalRef(jglobals);  

      ResultDecoder dec { this };
      auto off = dec.decode(region.get(), (jobjectArray) res);
      env()->DeleteGlobalRef(res);

      return off;
    }
};

struct SparkFunctionContext {
  RegionPtr region_;
  SparkEnv spark_env_;
  jobject jhadoop_conf_;

  SparkFunctionContext(RegionPtr region, jobject jhadoop_conf) :
  region_(region), spark_env_(), jhadoop_conf_(nullptr) {
    if (jhadoop_conf != nullptr) {
      jhadoop_conf_ = spark_env_.env()->NewGlobalRef(jhadoop_conf);
    }
  }

  ~SparkFunctionContext() {
    if (jhadoop_conf_    != nullptr) {
      spark_env_.env()->DeleteGlobalRef(jhadoop_conf_);
    }
  }
};

}

#endif