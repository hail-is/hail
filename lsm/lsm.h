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

class Level {
public:
  std::vector <File> files;
  const int max_size = 2;
  int index;
  std::filesystem::path level_directory;

  explicit Level(int _index, std::filesystem::path _directory) :
      index{_index}, level_directory{(std::filesystem::path) _directory / std::to_string(index)} {
    if (std::filesystem::exists(level_directory)) {
      std::cerr << "WARNING: " << level_directory << " already exists.";
    }
    std::filesystem::create_directory(level_directory);
  }

  int size();
  void add(std::map<int32_t, maybe_value> m);
  void add_file(File f);
  std::string next_file_path();
  File write_to_file(std::map<int32_t, maybe_value> m, std::string filename);
  std::map<int32_t, maybe_value> read_from_file(std::string filename);
  void read_to_map(std::string filename, std::map<int32_t, maybe_value> &m);
  File merge(File older_f, File newer_f);
};

class LSM {
  std::map<int32_t, maybe_value> m;
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
  Level& get_level(size_t index);
  void add_to_level(std::map<int32_t, maybe_value> m, size_t l_index);
  void dump_map();
};
