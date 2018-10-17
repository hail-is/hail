#ifndef HAIL_OBJECTARRAY_H
#define HAIL_OBJECTARRAY_H 1

#include <jni.h>
#include "hail/NativeObj.h"
#include <memory>
#include <vector>

namespace hail {

class ObjectArray : public NativeObj {
 public:
  std::vector<jobject> vec_;

 public:
  // Construct from a Scala Array[Object] passed through JNI as jobjectArray
  ObjectArray(JNIEnv* env, jobjectArray objects);
  
  ObjectArray(JNIEnv* env, jobject a0);
  
  ObjectArray(JNIEnv* env, jobject a0, jobject a1);
  
  ObjectArray(JNIEnv* env, jobject a0, jobject a1, jobject a2);
  
  ObjectArray(JNIEnv* env, jobject a0, jobject a1, jobject a2, jobject a3);

  ~ObjectArray();
  
  ObjectArray(const ObjectArray& b) = delete;
  ObjectArray& operator=(const ObjectArray& b) = delete;
  
  size_t size() const { return vec_.size(); }
  
  jobject at(ssize_t idx) const { return vec_[idx]; }
  
  jobject operator[](ssize_t idx) const { return vec_[idx]; }
};

using ObjectArrayPtr = std::shared_ptr<ObjectArray>;

class ObjectHolder : public NativeObj {
 public:
  ObjectArrayPtr objects_;

  ObjectHolder(ObjectArray* objects) :
    objects_(std::dynamic_pointer_cast<ObjectArray>(objects->shared_from_this())) {
  }
};

}

#endif
