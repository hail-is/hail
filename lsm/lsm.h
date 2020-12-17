#include <iostream>
#include <optional>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <fstream>
#include <limits>
#include <bitset>
#include <filesystem>

class BloomFilter {
  std::bitset<64> bset;
public:
  void insert_key(int32_t k);
  char contains_key(int32_t k);
};

class File {
public:
  std::string filename;
  int min, max;
  BloomFilter bloomFilter;
  File(std::string filename, BloomFilter bloomFilter, int min, int max) {
    this->filename = filename;
    this->bloomFilter = bloomFilter;
    this->min = min;
    this->max = max;
  }
};

class maybe_value {
public:
  int32_t v;
  char is_deleted;
  explicit maybe_value() : v{0}, is_deleted{0} {}
  explicit maybe_value(int32_t _v, char _deleted) : v{_v}, is_deleted{_deleted} {}
};

class LSM {
  std::map<int32_t, maybe_value> m;
  //std::vector<File> files;
  std::vector<Level> levels;
  std::filesystem::path directory;
public:
  explicit LSM(std::string _directory) :
    m{}, levels{}, directory{_directory} {
    if (std::filesystem::exists(directory)) {
      std::cerr << "WARNING: " << directory << " already exists.";
    }
    std::filesystem::create_directory(directory);
  }
  void put(int32_t k, int32_t v, char deleted = 0);
  std::optional<int32_t> get(int32_t k);
  std::vector<std::pair<int32_t, int32_t>> range(int32_t l, int32_t r);
  void del(int32_t k);
  &Level get_level(size_t index);
  //File write_to_file(std::string filename);
  //std::map<int32_t, maybe_value> read_from_file(std::string filename);
  void dump_map();
};
