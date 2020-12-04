#include <iostream>
#include <optional>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <fstream>

class LSM {
  std::map<int32_t, int32_t> m;
public:
  void put(int32_t k, int32_t v) {
    m.insert(std::make_pair(k,v));
  }
  std::optional<int32_t> get(int32_t k) {
    auto it = m.find(k);
    if (it != m.end()) {
        return it->second;
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

  int read_from_file(std::string filename) {
        m.clear();
        if (auto istrm = std::ifstream (filename, std::ios::binary)) {

            while (!istrm.eof()) {
                int k, v;
                istrm.read(reinterpret_cast<char*>(&k), sizeof k);
                istrm.read(reinterpret_cast<char*>(&v), sizeof v);
                m.insert(std::make_pair(k,v));
            }
        } else {
            std::cerr << "could not open " << filename << "\n";
            exit(3);
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
    default:
      std::cout << "unrecognized command " << line << std::endl;
      exit(4);
    }
  }
}
