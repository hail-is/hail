#ifndef HAIL_ENCODERTEST_H
#define HAIL_ENCODERTEST_H 1
#include <jni.h>

namespace hail {

long write_rows(jobject os, jobject it);

}

#endif