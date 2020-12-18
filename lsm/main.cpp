#include <iostream>
#include <string>
#include "lsm.h"

int main(int argc, const char ** argv) {
  if (argc != 1) {
    std::cerr << "USAGE: main" << std::endl;
    return 1;
  }

  LSM lsm{"db"};

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
