#include <iostream>
#include <optional>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <fstream>

class File {
  std::string filename;
  int min, max;
public:
  File(std::string filename) {
    this->filename = filename;
    std::vector<int32_t> keys = get_keys();
    this->min = *min_element(keys.begin(), keys.end());
    this->max = *max_element(keys.begin(), keys.end());
  }
  int get_min() {
    return min;
  }
  int get_max() {
    return max;
  }
  std::string get_name() {
    return filename;
  }
  std::vector<int32_t> get_keys() {
    std::vector<int32_t> keys;
    if (auto istrm = std::ifstream(filename, std::ios::binary)) {
      while (!istrm.eof()) {
        int k, v;
        istrm.read(reinterpret_cast<char *>(&k), sizeof k);
        istrm.read(reinterpret_cast<char *>(&v), sizeof v);
        keys.push_back(k);
      }
    } else {
      std::cerr << "could not open " << filename << "\n";
      exit(3);
    }
    return keys;
  }
};

class LSM {
  std::map<int32_t, int32_t> m;
  std::vector<File> files;
public:
  void put(int32_t k, int32_t v) {
    if (m.size() == 4) {
      std::string filename;
      if (files.empty()) {
        filename = '0';
      } else {
        filename = std::to_string(std::stoi(files.back().get_name()) + 1);
      }
      write_to_file(filename);
      files.push_back(File(filename));
      m.clear();
      m.insert(std::make_pair(k,v));
    } else{
      m.insert(std::make_pair(k,v));
    }
  }
  std::optional<int32_t> get(int32_t k) {
    auto it = m.find(k);
    if (it != m.end()) {
      return it->second;
    } else {
      // TODO: else if key doesn't exist check the files
//      auto it_b = this->files.begin();
//      auto it_e = this->files.end();
//      for (auto it=it_b; it!=it_e; ++it) {
      for (auto file : files) {
        if (k >= file.get_min() && k <= file.get_max()) {
          std::map<int32_t, int32_t> file_map = read_from_file(file.get_name());
          auto it_m = file_map.find(k);
          if (it_m != m.end()) {
            return it_m->second;
          }
        }
      }
    }
    return std::nullopt;
  }
  std::vector<std::pair<int32_t, int32_t>> range(int32_t l, int32_t r) {
    std::vector<std::pair<int32_t, int32_t>> res;
    auto it_l = m.lower_bound(l);
    auto it_u = m.lower_bound(r);
    for (auto it=it_l; it!=it_u; ++it) {
      res.push_back(std::make_pair(it->first, it->second));
    }
    return res;
  }
  void del(int32_t k) {
    m.erase(k);
  }
  int write_to_file(std::string filename) {
    std::ofstream ostrm(filename, std::ios::binary);
    for (auto const&x : m) {
      ostrm.write(reinterpret_cast<const char*>(&x.first), sizeof x.first);
      ostrm.write(reinterpret_cast<const char*>(&x.second), sizeof x.second);
    }
    return 0;
  }
  std::map<int32_t, int32_t> read_from_file(std::string filename) {
    //TODO: return map (don't modify actual map, create new and return)
    std::map<int32_t, int32_t> new_m;
    if (auto istrm = std::ifstream(filename, std::ios::binary)) {
      while (!istrm.eof()) {
        int k, v;
        istrm.read(reinterpret_cast<char *>(&k), sizeof k);
        istrm.read(reinterpret_cast<char *>(&v), sizeof v);
        new_m.insert(std::make_pair(k, v));
      }
    } else {
      std::cerr << "could not open " << filename << "\n";
      exit(3);
    }
    return new_m;
  }
  int dump_map() {
    for (auto const&x : m) {
      std::cout << x.first << "\n";
    }
    return 0;
  }
};


int main(int argc, const char ** argv) {
  if (argc != 1) {
    std::cerr << "USAGE: main" << std::endl;
    return 1;
  }

  LSM lsm;

  auto& in = std::cin;
  for (std::string line; std::getline(in, line);) {
    switch (line[0]) {
    case 'p': {
      size_t next;
      int k = std::stoi(line.substr(2), &next);
      int v = std::stoi(line.substr(2 + next + 1));
      lsm.put(k, v);
      break;
    }
    case 'g': {
      int k = std::stoi(line.substr(2));
      auto x = lsm.get(k);
      if (x) {
        std::cout << x.value();
      }
      std::cout << "\n";
      break;
    }
    case 'r': {
      size_t next;
      int l = std::stoi(line.substr(2), &next);
      int r = std::stoi(line.substr(2 + next + 1));
      auto x = lsm.range(l, r);
      auto begin = x.cbegin();
      auto end = x.cend();
      if (begin != end) {
        std::cout << begin->first << ":" << begin->second;
        ++begin;
        for (; begin != end; ++begin) {
          std::cout << " " << begin->first << ":" << begin->second;
        }
      }
      std::cout << "\n";
      break;
    }
    case 'd': {
      int k = std::stoi(line.substr(2));
      lsm.del(k);
      break;
    }
    case 'w': {
      std::string filename = line.substr(2);
      lsm.write_to_file(filename);
      break;
    }
    case 'R': {
      std::string filename = line.substr(2);
      lsm.read_from_file(filename);
      break;
    }
    case 'D': {
      lsm.dump_map();
      break;
    }
    default:
      std::cout << "unrecognized command " << line << std::endl;
      exit(4);
    }
  }
}
