#include <cstdio>
#include <sstream>

#include <hail/format.hpp>

namespace hail {

FormatStream::~FormatStream() {}

class StdFILEStream : public FormatStream {
  FILE *fp;

public:
  StdFILEStream(FILE *fp) : fp(fp) {}
  ~StdFILEStream();

  void putc(int c);
  void puts(const char *s);
};

/* don't close standard streams */
StdFILEStream::~StdFILEStream() {}

void
StdFILEStream::putc(int c) {
  fputc(c, fp);
}

void
StdFILEStream::puts(const char *s) {
  fputs(s, fp);
}

static StdFILEStream stdoutfs(stdout), stderrfs(stderr);

FormatStream &outs = stdoutfs, &errs = stderrfs;

template<class T> static inline void
write_with_iostream(FormatStream &s, T v) {
  std::ostringstream oss;
  oss << v;
  s.puts(oss.str().c_str());
}

void format1(FormatStream &s, signed char v) {
  write_with_iostream<signed char>(s, v);
}

void format1(FormatStream &s, unsigned char v) {
  write_with_iostream<unsigned char>(s, v);
}

void format1(FormatStream &s, short v) {
  write_with_iostream<short>(s, v);
}

void format1(FormatStream &s, unsigned short v) {
  write_with_iostream<unsigned short>(s, v);
}

void format1(FormatStream &s, int v) {
  write_with_iostream<int>(s, v);
}

void format1(FormatStream &s, unsigned int v) {
  write_with_iostream<unsigned int>(s, v);
}

void format1(FormatStream &s, long v) {
  write_with_iostream<long>(s, v);
}

void format1(FormatStream &s, unsigned long v) {
  write_with_iostream<unsigned long>(s, v);
}

void format1(FormatStream &s, const char *v) {
  s.puts(v);
}

}
