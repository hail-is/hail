#ifndef HAIL_PACKDECODER_H
#define HAIL_PACKDECODER_H 1

#include "hail/NativeObj.h"
#include "hail/ObjectArray.h"
#include "hail/Region.h"
#include <unordered_map>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

//#define MYDEBUG 1

//#define BIG_METHOD_INLINE 1

#ifdef BIG_METHOD_INLINE
# define MAYBE_INLINE inline
#else
# define MAYBE_INLINE
#endif

#define LIKELY(condition)   __builtin_expect(static_cast<bool>(condition), 1)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

namespace hail {

inline ssize_t round_up_align(ssize_t n, ssize_t align) {
  return ((n + (align-1)) & ~(align-1));
}
  
inline ssize_t missing_bytes(ssize_t nbits) {
  return ((nbits + 7) >> 3);
}
  
ssize_t elements_offset(ssize_t n, bool required, ssize_t align);

void set_all_missing(char* miss, ssize_t nbits);
  
void set_all_missing(std::vector<char>& missing_vec, ssize_t nbits);
  
void stretch_size(std::vector<char>& missing_vec, ssize_t minsize);

inline bool is_missing(char* missing_base, ssize_t idx) {
  return (bool)((missing_base[idx>>3] >> (idx&7)) & 0x1);
}

inline bool is_missing(const std::vector<char>& missing_vec, ssize_t idx) {
  return (bool)((missing_vec[idx>>3] >> (idx&7)) & 0x1);
}

class DecoderBase : public NativeObj {
private:
  static constexpr ssize_t kDefaultCapacity = (64*1024);
  static constexpr ssize_t kSentinelSize = 16;
public:
  ObjectArrayPtr input_;
  ssize_t capacity_;
  char*   buf_;
  ssize_t pos_;
  ssize_t size_;
  char*   rv_base_;
  char    tag_[8];
  
public:
  DecoderBase(ssize_t bufCapacity = 0);
  
  virtual ~DecoderBase();
  
  void set_input(ObjectArray* input);
  
  virtual int64_t get_field_offset(int field_size, const char* s);
  
  ssize_t read_to_end_of_block();

  // Returns -1 if input stream is exhausted, else 0x00-0xff  
  int64_t decode_one_byte();

  // Returns -1 if input stream is exhausted, else RegionValue addr  
  virtual int64_t decode_one_item(Region* region) = 0;
  
#ifdef MYDEBUG
  void hexify(char* out, ssize_t pos, char* p, ssize_t n);
#endif    
};

//
// DecoderId=0 fixed-size int/long
// DecoderId=1 variable-length LEB128
//
template<int DecoderId>
class PackDecoderBase : public DecoderBase {
 public:
  virtual ~PackDecoderBase() { }
  //
  // Decode methods for primitive types
  //
  bool decode_byte(int8_t* addr) {
    ssize_t pos = pos_;
    if (pos >= size_) return false;
    *addr = *(int8_t*)(buf_+pos);
    pos_ = pos+1;
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s A decode_byte() -> 0x%02x [%p]\n", tag_, (*addr) & 0xff, addr);
#endif
    return true;
  }
  
  bool skip_byte() {
    ssize_t pos = pos_+1;
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  bool decode_int(int32_t* addr) {
    ssize_t pos = pos_;
    if (pos+4 > size_) return false;
    *addr = *(int32_t*)&buf_[pos];
    pos_ = pos+4;
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s A decode_int() -> %d\n", tag_, *addr);
    char hex[256];
    hexify(hex, pos, buf_+pos, 4);
    fprintf(stderr, "%s", hex);
#endif
    return true;
  }
  
  bool skip_int() {
    ssize_t pos = pos_+4;
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  bool decode_length(ssize_t* addr) {
    int32_t len = 0;
    if (!decode_int(&len)) return false;
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s A decode_length() -> %d\n", tag_, len);
#endif
    *addr = (ssize_t)len;
    return true;
  }
  
  bool decode_long(int64_t* addr) {
    ssize_t pos = pos_;
    if (pos+8 > size_) return false;
    *addr = *(int64_t*)&buf_[pos];
    pos_ = pos+8;
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s A decode_long() -> %ld\n", tag_, (long)*addr);
    char hex[256];
    hexify(hex, pos, buf_+pos, 8);
    fprintf(stderr, "%s", hex);
#endif
    return true;
  }
  
  bool skip_long() {
    ssize_t pos = pos_+8;
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  bool decode_float(float* addr) {
    ssize_t pos = pos_;
    if (pos+4 > size_) return false;
    *addr = *(float*)(buf_+pos);
    pos_ = (pos+4);
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s A decode_float() -> %12e\n", tag_, (double)*addr);
#endif
    return true;
  }
  
  bool skip_float() {
    ssize_t pos = pos_+4;
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  bool decode_double(double* addr) {
    ssize_t pos = pos_;
    if (pos+8 > size_) return false;
    *addr = *(double*)(buf_+pos);
    pos_ = (pos+8);
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s A decode_double() -> %12e\n", tag_, *addr);
#endif
    return true;
  }
  
  bool skip_double() {
    ssize_t pos = pos_+8;
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  ssize_t decode_bytes(char* addr, ssize_t n) {
    ssize_t pos = pos_;
    ssize_t ngot = (size_ - pos);
    if (ngot > n) ngot = n;
    if (ngot > 0) {
      memcpy(addr, buf_+pos, ngot);
      pos_ = (pos + ngot);
    }
#ifdef MYDEBUG
    char hex[256];
    hexify(hex, pos, buf_+pos, (ngot < 32) ? ngot : 32);
    fprintf(stderr, "DEBUG: %s A decode_bytes(%ld) -> %ld\n", tag_, n, ngot);
    fprintf(stderr, "%s", hex);
#endif
    return ngot;
  }
  
  ssize_t skip_bytes(ssize_t n) {
    ssize_t pos = pos_;
    if (n > size_-pos) n = size_-pos;
    pos_ = pos + n;
    return n;
  }
};

template<>
class PackDecoderBase<1> : public DecoderBase {
 public:
  virtual ~PackDecoderBase() { }
  //
  // Decode methods for primitive types
  //
  bool decode_byte(int8_t* addr) {
    ssize_t pos = pos_;
    if (pos >= size_) return false;
    *addr = *(int8_t*)(buf_+pos);
    pos_ = (pos+1);
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s B decode_byte() -> 0x%02x [%p]\n", tag_, (*addr) & 0xff, addr);
#endif
    return true;
  }
  
  bool skip_byte() {
    if (pos_ >= size_) return false;
    pos_ += 1;
    return true;
  }
  
  bool decode_int_slow(int32_t* addr);
  
  bool decode_int(int32_t* addr) {
    ssize_t pos = pos_;
    int32_t b = *(int8_t*)&buf_[pos];
    if (LIKELY(b >= 0)) { // fast path: not sentinel, one-byte encoding
      *addr = b;
      pos_ = pos+1;
      return true;
    }
    return decode_int_slow(addr);
  }
  
  bool skip_int() {
    ssize_t pos = pos_;
    int val = 0;
    for (int shift = 0;; shift += 7) {
      if (pos >= size_) return false;
      int b = buf_[pos++];
      val |= ((b & 0x7f) << shift);
      if ((b & 0x80) == 0) break;
    }
    pos_ = pos;
    return true;
  }
  
  bool decode_length(ssize_t* addr) {
    int32_t len = 0;
    if (!decode_int(&len)) return false;
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s B decode_length() -> %d\n", tag_, len);
#endif
    *addr = (ssize_t)len;
    return true;
  }
  
  bool decode_long(int64_t* addr);

  bool skip_long() {
    ssize_t pos = pos_;
    do {
      if (pos >= size_) return false;
    } while ((buf_[pos++] & 0x80) != 0);
    pos_ = pos;
    return true;
  }
  
  bool decode_float(float* addr) {
    ssize_t pos = pos_;
    if (pos+4 > size_) return false;
    *addr = *(float*)(buf_+pos);
    pos_ = (pos+4);
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s B decode_float() -> %12e\n", tag_, (double)*addr);
#endif
    return true;
  }
  
  bool skip_float() {
    ssize_t pos = pos_ + sizeof(float);
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  bool decode_double(double* addr) {
    ssize_t pos = pos_;
    if (pos+8 > size_) return false;
    *addr = *(double*)(buf_+pos);
    pos_ = (pos+8);
#ifdef MYDEBUG
    fprintf(stderr, "DEBUG: %s B decode_double() -> %12e\n", tag_, *addr);
#endif
    return true;
  }
  
  bool skip_double() {
    ssize_t pos = pos_ + sizeof(double);
    if (pos > size_) return false;
    pos_ = pos;
    return true;
  }
  
  ssize_t decode_bytes(char* addr, ssize_t n);

  ssize_t skip_bytes(ssize_t n) {
    ssize_t pos = pos_;
    if (n > size_-pos) n = size_-pos;
    pos_ = pos + n;
    return n;
  }
};

MAYBE_INLINE bool PackDecoderBase<1>::decode_int_slow(int32_t* addr) {
  ssize_t pos = pos_;
  int val = 0;
  for (int shift = 0;; shift += 7) {
    if (pos >= size_) return false;
    int b = buf_[pos++];
    val |= ((b & 0x7f) << shift);
    if ((b & 0x80) == 0) break;
  }
  *addr = val;
#ifdef MYDEBUG
  fprintf(stderr, "DEBUG: %s B decode_int() -> %d\n", tag_, val);
  char hex[256];
  hexify(hex, pos_, buf_+pos_, pos-pos_);
  fprintf(stderr, "%s", hex);
#endif
  pos_ = pos;
  return true;
}

MAYBE_INLINE bool PackDecoderBase<1>::decode_long(int64_t* addr) {
  ssize_t pos = pos_;
  ssize_t val = 0;
  for (int shift = 0;; shift += 7) {
    if (pos >= size_) return false;
    ssize_t b = buf_[pos++];
    val |= ((b & 0x7f) << shift);
    if ((b & 0x80) == 0) break;
  }
  *addr = val;
#ifdef MYDEBUG
  fprintf(stderr, "DEBUG: %s B decode_long() -> %ld\n", tag_, val);
  char hex[256];
  hexify(hex, pos_, buf_+pos_, pos-pos_);
  fprintf(stderr, "%s", hex);
#endif
  pos_ = pos;
  return true;
}
  
MAYBE_INLINE ssize_t PackDecoderBase<1>::decode_bytes(char* addr, ssize_t n) {
  ssize_t pos = pos_;
  ssize_t ngot = (size_ - pos);
  if (ngot > n) ngot = n;
  if (ngot > 0) {
    memcpy(addr, buf_+pos, ngot);
    pos_ = (pos + ngot);
  }
#ifdef MYDEBUG
  char hex[256];
  hexify(hex, pos, buf_+pos, (ngot < 32) ? ngot : 32);
  fprintf(stderr, "DEBUG: %s B decode_bytes(%ld) -> %ld\n", tag_, n, ngot);
  fprintf(stderr, "%s", hex);
#endif
  return ngot;
}

} // end hail

#endif
