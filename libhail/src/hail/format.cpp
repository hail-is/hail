#include <cassert>
#include <cerrno>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <sstream>

#include <hail/format.hpp>

namespace hail {

FormatStream::~FormatStream() {}

StringFormatStream::StringFormatStream() {}
StringFormatStream::~StringFormatStream() {}

void
StringFormatStream::write(const char *p, size_t n) {
  const char *end = p + n;
  for (; p < end; ++p)
    contents.push_back(*p);
}

void
StringFormatStream::putc(int c) {
  contents.push_back(c);
}

void
StringFormatStream::puts(const char *s) {
  char c = *s;
  while (c) {
    contents.push_back(c);
    c = *(++s);
  }
}

std::string
StringFormatStream::get_contents() {
  return std::move(contents);
}

class StdFILEStream : public FormatStream {
  FILE *fp;

public:
  StdFILEStream(FILE *fp) : fp(fp) {}
  ~StdFILEStream();

  void write(const char *p, size_t n) override;
  void putc(int c) override;
  void puts(const char *s) override;
};

/* don't close standard streams */
StdFILEStream::~StdFILEStream() {}

void
StdFILEStream::write(const char *p, size_t n) {
  int rc = fwrite(p, n, 1, fp);
  if (rc != 1)
    throw std::system_error(errno, std::generic_category(), "in fwrite");
}

void
StdFILEStream::putc(int c) {
  int rc = fputc(c, fp);
  if (rc != c)
    throw std::system_error(errno, std::generic_category(), "in fputc");
}

void
StdFILEStream::puts(const char *s) {
  int rc = fputs(s, fp);
  if (rc < 0)
    throw std::system_error(errno, std::generic_category(), "in fputc");
}

static StdFILEStream stdoutfs(stdout), stderrfs(stderr);

FormatStream &outs = stdoutfs, &errs = stderrfs;

template<class T> static inline void
write_with_iostream(FormatStream &s, T v) {
  std::ostringstream oss;
  oss << v;
  s.puts(oss.str().c_str());
}

void
format1(FormatStream &s, bool v) {
  s.puts(v ? "true" : "false");
}

void
format1(FormatStream &s, signed char v) {
  write_with_iostream<signed char>(s, v);
}

void
format1(FormatStream &s, unsigned char v) {
  write_with_iostream<unsigned char>(s, v);
}

void
format1(FormatStream &s, short v) {
  write_with_iostream<short>(s, v);
}

void
format1(FormatStream &s, unsigned short v) {
  write_with_iostream<unsigned short>(s, v);
}

void
format1(FormatStream &s, int v) {
  write_with_iostream<int>(s, v);
}

void
format1(FormatStream &s, unsigned int v) {
  write_with_iostream<unsigned int>(s, v);
}

void
format1(FormatStream &s, long v) {
  write_with_iostream<long>(s, v);
}

void
format1(FormatStream &s, unsigned long v) {
  write_with_iostream<unsigned long>(s, v);
}

void
format1(FormatStream &s, long long v) {
  write_with_iostream<long long>(s, v);
}

void
format1(FormatStream &s, unsigned long long v) {
  write_with_iostream<unsigned long long>(s, v);
}

void
format1(FormatStream &s, float v) {
  write_with_iostream<unsigned long>(s, v);
}

void
format1(FormatStream &s, double v) {
  write_with_iostream<double>(s, v);
}

void
format1(FormatStream &s, const char *v) {
  s.puts(v);
}

void
format1(FormatStream &s, const std::string &t) {
  s.puts(t.c_str());
}

void
format1(FormatStream &s, FormatAddress v) {
  std::ostringstream oss;
  oss << std::hex << std::showbase << std::internal << std::setw(16) << std::setfill('0')
      << reinterpret_cast<uintptr_t>(v.p);
  s.puts(oss.str().c_str());
}

void
format1(FormatStream &s, Indent v) {
  assert(v.indent >= 0);
  for (int i = 0; i < v.indent; ++i)
    s.putc(' ');
}

}
