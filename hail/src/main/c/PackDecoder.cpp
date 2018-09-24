#include "hail/PackDecoder.h"
#include "hail/Upcalls.h"
#include <cassert>

namespace hail {

ssize_t elements_offset(ssize_t n, bool required, ssize_t align) {
  return round_up_align(sizeof(int32_t) + (required ? 0 : missing_bytes(n)), align);
}

void set_all_missing(char* miss, ssize_t nbits) {
  memset(miss, 0xff, (nbits+7)>>3);
  int partial = (nbits & 0x7);
  if (partial != 0) miss[nbits>>3] = (1<<partial)-1;
}

void set_all_missing(std::vector<char>& missing_vec, ssize_t nbits) {
  ssize_t nbytes = ((nbits+7)>>3);
  if (missing_vec.size() < (size_t)nbytes) missing_vec.resize(nbytes);
  memset(&missing_vec[0], 0xff, nbytes);
  int partial = (nbits & 0x7);
  if (partial != 0) missing_vec[nbits>>3] = (1<<partial)-1;
}

void stretch_size(std::vector<char>& missing_vec, ssize_t minsize) {
  if (ssize(missing_vec) < minsize) missing_vec.resize(minsize);
}

DecoderBase::DecoderBase(ssize_t bufCapacity) :
  input_(),
  capacity_((bufCapacity > 0) ? bufCapacity : kDefaultCapacity),
  buf_((char*)malloc(capacity_ + ((kSentinelSize+0x3f) & ~0x3f))),
  pos_(0),
  size_(0),
  rv_base_(nullptr) {
  sprintf(tag_, "%04lx", ((long)this & 0xffff) | 0x8000);
}

DecoderBase::~DecoderBase() {
  auto buf = buf_;
  buf_ = nullptr;
  if (buf) free(buf);
}

void DecoderBase::set_input(ObjectArray* input) {
  input_ = std::dynamic_pointer_cast<ObjectArray>(input->shared_from_this());
}

int64_t DecoderBase::get_field_offset(int field_size, const char* s) {
  auto zeroObj = reinterpret_cast<DecoderBase*>(0L);
  if (!strcmp(s, "capacity_")) return (int64_t)&zeroObj->capacity_;
  if (!strcmp(s, "buf_"))      return (int64_t)&zeroObj->buf_;
  if (!strcmp(s, "pos_"))      return (int64_t)&zeroObj->pos_;
  if (!strcmp(s, "size_"))     return (int64_t)&zeroObj->size_;
  if (!strcmp(s, "rv_base_"))  return (int64_t)&zeroObj->rv_base_;
  return -1;
}

#ifdef MYDEBUG
void DecoderBase::hexify(char* out, ssize_t pos, char* p, ssize_t n) {    
  for (int j = 0; j < n; j += 8) {
    sprintf(out, "[%4ld] ", pos+j);
    out += strlen(out);
    for (int k = 0; k < 8; ++k) {
      if (j+k >= n) {
        *out++ = ' ';
        *out++ = ' ';
      } else {
        int c = (j+k < n) ? (p[j+k] & 0xff) : ' ';
        int nibble = (c>>4) & 0xff;
        *out++ = ((nibble < 10) ? '0'+nibble : 'a'+nibble-10);
        nibble = (c & 0xf);
        *out++ = ((nibble < 10) ? '0'+nibble : 'a'+nibble-10);
      }
      *out++ = ' ';
    }
    *out++ = ' ';
    for (int k = 0; k < 8; ++k) {
      int c = (j+k < n) ? (p[j+k] & 0xff) : ' ';
      *out++ = ((' ' <= c) && (c <= '~')) ? c : '.';
    }
    *out++ = '\n';
  }
  *out++ = 0;
}
#endif

ssize_t DecoderBase::read_to_end_of_block() {
  assert(size_ >= 0);
  assert(size_ <= capacity_);
  assert(pos_ >= 0);
  assert(pos_ <= size_+1);
  auto remnant = (size_ - pos_);
  if (remnant < 0) {
    return -1;
  }
  if (remnant > 0) {
    memcpy(buf_, buf_+pos_, remnant);
  }
  pos_ = 0;
  size_ = remnant;
  int32_t chunk = (capacity_ - size_);
  UpcallEnv up;
  // The unused Array[Byte] parameter gets (jbyteArray)0 which is Scala/Java nil
  int32_t rc = up.InputBuffer_readToEndOfBlock(input_->at(0), buf_+size_, (jbyteArray)0,
                                               0, chunk);
  assert(rc <= chunk);
  if (rc < 0) {
    pos_ = (size_ + 1); // (pos > size) means end-of-file
    return -1;
  } else {
    size_ += rc;
    // buf is oversized with space for a sentinel to speed up one-byte-int decoding
    memset(buf_+size_, 0xff, kSentinelSize-1);
    buf_[size_+kSentinelSize-1] = 0x00; // terminator for LEB128 loop
    return rc;
  }
}

int64_t DecoderBase::decode_one_byte() {
  ssize_t avail = (size_ - pos_);
  if (avail <= 0) {
    if ((avail < 0) || (read_to_end_of_block() <= 0)) {
      return -1;
    }
  }
  int64_t result = (buf_[pos_++] & 0xff);
  return result;
}

} // end hail
