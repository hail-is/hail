#ifndef HAIL_REGION_H
#define HAIL_REGION_H 1

#include "hail/hail.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include "hail/Region2.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>

namespace hail {

class ScalaRegionPool : public NativeObj {
  RegionPool pool_{};
  public:
    class Region : public NativeObj {
      private:
        RegionPtr region_;

      public:
        Region(ScalaRegionPool * pool) : region_(pool->pool_.get_region()) { }
        void clear() {
          auto r2 = region_->get_region();
          region_ = nullptr;
          region_ = std::move(r2);
        }
        inline void align(size_t alignment) {
          region_->align(alignment);
        }
        inline char * allocate(size_t alignment, size_t n) { return region_->allocate(alignment, n); }
        inline char * allocate(size_t n) { return region_->allocate(n); }

        virtual const char* get_class_name() { return "Region"; }
        virtual ~Region() {
          region_ = nullptr;
        }
    };

    inline std::shared_ptr<Region> get_region() {
      return std::make_shared<Region>(this);
    }
    virtual const char* get_class_name() { return "RegionPool"; }
};

using Region = ScalaRegionPool::Region;

#define REGIONMETHOD(rtype, scala_class, scala_method) \
  extern "C" __attribute__((visibility("default"))) \
    rtype Java_is_hail_annotations_##scala_class##_##scala_method

REGIONMETHOD(void, RegionPool, nativeCtor)(
  JNIEnv* env,
  jobject thisJ
) {
  NativeObjPtr ptr = std::make_shared<ScalaRegionPool>();
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, nativeCtor)(
  JNIEnv* env,
  jobject thisJ,
  jobject poolJ
) {
  auto pool = static_cast<ScalaRegionPool*>(get_from_NativePtr(env, poolJ));
  NativeObjPtr ptr = std::make_shared<ScalaRegionPool::Region>(pool);
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, clearButKeepMem)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  r->clear();
}

REGIONMETHOD(void, Region, nativeAlign)(
  JNIEnv* env,
  jobject thisJ,
  jlong a
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  r->align(a);
}

REGIONMETHOD(jlong, Region, nativeAlignAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong a,
  jlong n
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<jlong>(r->allocate((size_t)a, (size_t)n));
}

REGIONMETHOD(jlong, Region, nativeAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong n
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<jlong>(r->allocate((size_t)n));
}

} // end hail

#endif
