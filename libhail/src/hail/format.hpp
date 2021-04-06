#ifndef HAIL_FORMAT_HPP
#define HAIL_FORMAT_HPP 1

#include <string>
#include <cstddef>
#include <utility>

namespace hail {

class FormatStream {
public:
  virtual ~FormatStream();

  virtual void write(const char *p, size_t n) = 0;
  virtual void putc(int c) = 0;
  virtual void puts(const char *s) = 0;
};

class StringFormatStream : public FormatStream {
  std::string contents;

public:
  StringFormatStream();
  ~StringFormatStream();

  void write(const char *p, size_t n) override;
  void putc(int c) override;
  void puts(const char *s) override;

  std::string get_contents();
};

extern FormatStream &outs, &errs;

extern void format1(FormatStream &s, bool v);
extern void format1(FormatStream &s, signed char v);
extern void format1(FormatStream &s, unsigned char v);
extern void format1(FormatStream &s, short v);
extern void format1(FormatStream &s, unsigned short v);
extern void format1(FormatStream &s, int v);
extern void format1(FormatStream &s, unsigned int v);
extern void format1(FormatStream &s, long v);
extern void format1(FormatStream &s, long long v);
extern void format1(FormatStream &s, unsigned long v);
extern void format1(FormatStream &s, unsigned long long v);
extern void format1(FormatStream &s, float v);
extern void format1(FormatStream &s, double v);
extern void format1(FormatStream &s, const char *v);
extern void format1(FormatStream &s, const std::string &t);

class FormatAddress {
public:
  const void *p;
  FormatAddress(const void *p) : p(p) {}
};

extern void format1(FormatStream &s, FormatAddress v);

class Indent {
public:
  int indent;
  Indent(int indent) : indent(indent) {}
};

extern void format1(FormatStream &s, Indent v);

template<typename... Args> inline void
format(FormatStream &s, Args &&... args) {
  (format1(s, std::forward<Args>(args)),...);
}

template<typename... Args> inline std::string
render(Args &&... args) {
  StringFormatStream s;
  format(s, std::forward<Args>(args)...);
  return std::move(s.get_contents());
}

template<typename... Args> inline void
print(Args &&... args) {
  format(outs, std::forward<Args>(args)...);
  outs.putc('\n');
}

}
#endif
