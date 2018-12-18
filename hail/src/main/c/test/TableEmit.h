#ifndef HAIL_TEST_TABLEEMIT_H
#define HAIL_TEST_TABLEEMIT_H 1

#include "hail/table/PartitionContext.h"
#include <vector>
#include <numeric>
#include <cstring>
#include <string>

namespace hail {

struct TestStringImplementation {
  std::vector<std::string> rows_;
  std::string globals_;

  template <typename Mapper>
  std::vector<std::string> map(Mapper mapper) {
    std::vector<std::string> result;
    for (auto s : rows_) { result.push_back(mapper(globals_, s)); }
    return result;
  }

  template <typename Filter>
  std::vector<std::string> filter(Filter filterer) {
    std::vector<std::string> result;
    for (auto s : rows_) {
      if (filterer(globals_, s)) {
        result.push_back(s);
      }
    }
    return result;
  }

  template <typename LenF,typename GetIF>
  std::vector<std::string> explode(LenF len_f, GetIF get_i) {
    std::vector<std::string> result;
    for (auto s : rows_) {
      for (int i=0; i < len_f(s); ++i) {
        result.push_back(get_i(s, i));
      }
    }
    return result;
  }

  TestStringImplementation(std::vector<std::string> rows, std::string globals) :
  rows_(rows), globals_(globals) { }
};

struct TestStringDecoder {
  std::vector<std::string> rows_;
  size_t i_ = 0;

  char decode_byte() { return (i_ < rows_.size()) ? 1 : 0; }

  char const * decode_row(Region * region) {
    int len = (int) rows_[i_].size();
    auto off = region->allocate(len + sizeof(int));
    *reinterpret_cast<int *>(off) = len;
    memcpy(off + sizeof(int), rows_[i_].c_str(), len);
    ++i_;
    return off;
  }

  explicit TestStringDecoder(std::vector<std::string> strings) :
  rows_(strings) { }
};

struct TestStringEncoder {
  std::vector<std::string> rows_;
  unsigned int size_ = 0;
  bool at_end_ = false;

  void encode_byte(char b) {
    if (b == 0) { at_end_ = true; } else { ++size_; }
  }

  void encode_row(char const * row) {
    int len = *reinterpret_cast<int const *>(row);
    if (at_end_ || (size_ != (rows_.size() + 1))) { size_ = -1; }
    rows_.emplace_back(row + sizeof(int), len);
  }

  void flush() { }

//  TestStringEncoder() : TestStringEncoder(std::make_shared<std::vector<std::string>>()) { }
};

struct AppendString {
  char const * operator()(Region * region, const char * globals, const char * row) {
    int len = *reinterpret_cast<const int *>(row);
    int lenglob = *reinterpret_cast<const int *>(globals);
    char * off = region->allocate(len + lenglob + sizeof(int));
    *reinterpret_cast<int *>(off) = (len + lenglob);
    memcpy(off + sizeof(int), row + sizeof(int), len);
    memcpy(off + sizeof(int) + len, globals + sizeof(int), lenglob);
    return off;
  }
};

struct FilterString {
  bool filter(const char * globals, const char * row) {
    int len = *reinterpret_cast<const int *>(row);
    int lenglob = *reinterpret_cast<const int *>(globals);
    return len == lenglob;
  }
  bool operator()(Region * region, const char * globals, const char * row) {
    return filter(globals, row);
  }
};


struct ExplodeToChars {
  int len(Region * region, const char * value) {
    return *reinterpret_cast<const int *>(value);
  }

  const char * operator()(Region * region, const char * row, int i) {
    auto off = region->allocate(sizeof(int) + 1);
    *reinterpret_cast<int *>(off) = 1;
    *(off + sizeof(int)) = row[i + sizeof(int)];
    return off;
  }
};

}

#endif