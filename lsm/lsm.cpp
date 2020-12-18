#include "lsm.h"
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <fstream>
#include <limits>
#include <bitset>
#include <filesystem>
#include "MurmurHash3.h"

void BloomFilter::insert_key(int32_t k) {
  uint32_t seed = 1;
  uint32_t key_hash;
  MurmurHash3_x86_32(&k, sizeof k, seed, &key_hash);
  auto last_d = key_hash % 10;
  bset[last_d] = 1;
}
char BloomFilter::contains_key(int32_t k) {
  uint32_t seed = 1;
  uint32_t key_hash;
  MurmurHash3_x86_32(&k, sizeof k, seed, &key_hash);
  auto last_d = key_hash % 10;
  return bset[last_d];
}

int Level::size() {
  return files.size();
}
void Level::add_file(File f) {
  files.push_back(f);
}
std::string Level::next_file_path() {
  return level_directory / std::to_string(files.size());
}
File Level::write_to_file(std::map<int32_t, maybe_value> m, std::string filename) {
  std::ofstream ostrm(filename, std::ios::binary);
  BloomFilter bloomFilter;
  int min = std::numeric_limits<int>::max();
  int max = std::numeric_limits<int>::lowest();
  for (auto const&x : m) {
    bloomFilter.insert_key(x.first);
    ostrm.write(reinterpret_cast<const char*>(&x.first), sizeof x.first);
    ostrm.write(reinterpret_cast<const char*>(&x.second.v), sizeof x.second.v);
    ostrm.write(reinterpret_cast<const char*>(&x.second.is_deleted), sizeof x.second.is_deleted);
    if (x.first > max) {max = x.first;}
    if (x.first < min) {min = x.first;}
  }
  std::cerr << "write to file " << filename << "\n";
  return File(filename, bloomFilter, min, max);
}
std::map<int32_t, maybe_value> Level::read_from_file(std::string filename) {
  std::map<int32_t, maybe_value> new_m;
  read_to_map(filename, new_m);
  return new_m;
}
void Level::read_to_map(std::string filename, std::map<int32_t, maybe_value> &m) {
  if (auto istrm = std::ifstream(filename, std::ios::binary)) {
    int k;
    while (istrm.read(reinterpret_cast<char *>(&k), sizeof k)) {
      int v;
      char d;
      istrm.read(reinterpret_cast<char *>(&v), sizeof v);
      istrm.read(reinterpret_cast<char *>(&d), sizeof d);
      m.insert_or_assign(k, maybe_value(v,d));
    }
  } else {
    std::cerr << "could not open " << filename << "\n";
    exit(3);
  }
  std::cerr << "read file " << filename << " to map" << "\n";
}
File Level::merge(File older_f, File newer_f) {
  std::map<int32_t, maybe_value> m;
  read_to_map(older_f.filename, m);
  read_to_map(newer_f.filename, m);
  std::cerr << "merge files " << older_f.filename << "and" << newer_f.filename << " to new file " << "\n";
  return write_to_file(m, next_file_path());
}
void Level::add(std::map<int32_t, maybe_value> m) {
  File f = write_to_file(m, next_file_path()); //TODO & ?
  add_file(f);
}

void LSM::add_to_level(std::map<int32_t, maybe_value> m, size_t l_index) {
  Level& level = get_level(l_index); //TODO
  if (l_index >= levels.size()) {
    //get_level(l_index).add(m);
    level.add(m);
  } else if (level.size() + 1 >= level.max_size) {
    assert(level.max_size == 2);
    File merged_f = level.merge(level.files.back(),
                                          level.write_to_file(m, level.next_file_path()));
    std::map<int32_t, maybe_value> merged_m;
    level.read_to_map(merged_f.filename, merged_m);
    add_to_level(merged_m, l_index + 1);
    std::filesystem::path file_path = get_level(l_index).files.back().filename;
    get_level(l_index).files.pop_back(); //TODO: why does this error heap-used-after-free if I replace `get_level(l_index)` with `level`?
    std::filesystem::remove(file_path);
  } else {
    level.add(m);
  }
}
Level& LSM::get_level(size_t index) {
  if (index > levels.size()) {
    std::cerr << "skipped a level!?";
    exit(1);
  } else if (index == levels.size()) {
    levels.push_back(Level(index, directory));
    return levels.back();
  } else {
    return levels[index];
  }
}
void LSM::put(int32_t k, int32_t v, char deleted) {
  if (m.size() >= 4) {
    add_to_level(m, 0);
    //get_level(0).add(m);
    m.clear();
  }
  m.insert_or_assign(k,maybe_value(v, deleted));
}
std::optional<int32_t> LSM::get(int32_t k) {
  auto it = m.find(k);

  if (it != m.end()) {
    if(!it->second.is_deleted) {
      return it->second.v;
    } else {
      return std::nullopt;
    }
  } else {
    for(unsigned i = levels.size() - 1; levels.size() > i; --i) {
      Level& level = get_level(i);
      for (auto j = level.files.rbegin(); j != level.files.rend(); ++j) {
        File& file = *j;
        if (file.bloomFilter.contains_key(k) && k >= file.min && k <= file.max) {
          std::map <int32_t, maybe_value> file_map = level.read_from_file(file.filename);
          auto it_m = file_map.find(k);
          if (it_m != file_map.end()) {
            if (!it_m->second.is_deleted) {
              return it_m->second.v;
            } else {
              return std::nullopt;
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}
std::vector<std::pair<int32_t, int32_t>> LSM::range(int32_t l, int32_t r) {
  std::vector<std::pair<int32_t, int32_t>> res;
  std::map<int32_t,int32_t>  res_map;

  for (auto &level : levels) {
    for (auto &file : level.files) {
      if (r >= file.min && l <= file.max) {
        std::map <int32_t, maybe_value> file_map = level.read_from_file(file.filename);
        auto it_ml = file_map.lower_bound(l);
        auto it_mu = file_map.lower_bound(r);
        for (auto it = it_ml; it != it_mu; ++it) {
          if (!it->second.is_deleted) {
            res_map.insert_or_assign(it->first, it->second.v);
          } else {
            res_map.erase(it->first);
          }
        }
      }
    }
  }

  auto it_l = m.lower_bound(l);
  auto it_u = m.lower_bound(r);
  for (auto it=it_l; it!=it_u; ++it) {
    if (!it->second.is_deleted) {
      res_map.insert_or_assign(it->first, it->second.v);
    } else {
      res_map.erase(it->first);
    }
  }

  auto rit_l = res_map.lower_bound(l);
  auto rit_u = res_map.lower_bound(r);
  for (auto it=rit_l; it!=rit_u; ++it) {
    res.push_back(std::make_pair(it->first, it->second));
  }

  return res;
}
void LSM::del(int32_t k) {
  put(k, 0, 1);
}
void LSM::dump_map() {
  for (auto const&x : m) {
    std::cout << x.first << "\n";
  }
}
