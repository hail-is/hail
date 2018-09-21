#include "hail/hstring.h"
#include <cstdlib>
#include <cstring>

namespace hail {

void hstring::init(const char* s, size_t n) {
  capacity_ = 0;
  length_ = 0;
  buf_ = nullptr;
  if (n > 0) {
    reserve(n);
    memcpy(buf_, s, n);
    buf_[n] = 0;
    length_ = n;
  }
}

void hstring::assign(const char* s, size_t n) {
  if (n > 0) {
    if (n > capacity_) reserve(n);
    memcpy(buf_, s, n);
    buf_[n] = 0;
  }
  length_ = n;
}

hstring::hstring(const char* s, size_t n) { init(s, n); }

hstring::hstring(const char* s) { init(s, strlen(s)); }

hstring::hstring(const hstring& b) { init(b.data(), b.length()); }

hstring::hstring(hstring&& b) :
  capacity_(b.capacity_),
  length_(b.length_),
  buf_(b.buf_) {
  b.capacity_ = 0;
  b.length_ = 0;
  b.buf_ = nullptr;
}

hstring::~hstring() {
  if (buf_) free(buf_);
}

hstring& hstring::operator=(const char* s) { assign(s, strlen(s)); return *this; }

hstring& hstring::operator=(const hstring& b) { assign(b.buf_, b.length_); return *this; }

hstring& hstring::operator=(hstring&& b) {
  capacity_ = b.capacity_;
  length_ = b.length_;
  if (buf_) free(buf_);
  buf_ = b.buf_;
  b.capacity_ = 0;
  b.length_ = 0;
  b.buf_ = nullptr;
  return *this;
}

bool hstring::operator==(const hstring& b) const { return (strcmp(this->c_str(), b.c_str()) == 0); }

bool hstring::operator!=(const hstring& b) const { return (strcmp(this->c_str(), b.c_str()) != 0); }

bool hstring::operator<(const hstring& b) const { return (strcmp(this->c_str(), b.c_str()) < 0); }

void hstring::reserve(size_t min_cap) {
  if (min_cap <= capacity_) return;
  if (min_cap < length_) min_cap = length_;
  size_t cap1 = (capacity_ ? capacity_ : kMinCapacity) + 1;
  while (cap1 <= min_cap) cap1 <<= 1;
  char* new_buf = (char*)malloc(cap1);
  if (length_ > 0) memcpy(new_buf, buf_, length_);
  if (buf_) free(buf_);
  new_buf[length_] = 0;
  capacity_ = (cap1 - 1);
  buf_ = new_buf;
}

hstring hstring::operator+(const hstring& b) const {
  hstring result;
  auto a_len = this->length_;
  auto b_len = b.length_;
  auto new_len = (a_len + b_len);
  result.reserve(new_len);
  if (a_len > 0) memcpy(result.buf_, this->buf_, a_len);
  if (b_len > 0) memcpy(result.buf_+a_len, b.buf_, b_len);
  result.buf_[new_len] = 0;
  result.length_ = new_len;
  return result;
}

void hstring::append(const char* s, size_t n) {
  if (n == 0) return;
  auto new_len = (length_ + n);
  if (new_len > capacity_) reserve(new_len);
  memcpy(buf_+length_, s, n);
  buf_[new_len] = 0;
  length_ = new_len;
}

hstringstream& hstringstream::operator<<(const hstring& b) {
  str_.append(b.data(), b.length());
  return *this;
}

hstringstream& hstringstream::operator<<(const char* s) {
  str_.append(s, strlen(s));
  return *this;
}

hstringstream& hstringstream::operator<<(uint64_t n) {
  // Upper bound on length is 6 bytes per 16 bits = 24 bytes
  static const size_t kBufLen = 32;
  char buf[kBufLen];
  char* out = &buf[kBufLen];
  *--out = 0;
  do {
    *--out = '0'+(n % 10);
    n /= 10;
  } while (n != 0);
  str_.append(out, strlen(out));
  return *this;
}

hstringstream& hstringstream::operator<<(int64_t n) {
  uint64_t mag = n;
  if (n < 0) {
    *this << '-';
    mag = -n;
  }
  return *this << mag;
}

hstringstream& hstringstream::operator<<(char c) {
  str_.append(&c, 1);
  return *this;
}

} // end hail
