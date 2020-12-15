#include "lsm.h"

void BloomFilter::insert_key(int32_t k) {
  auto key_hash = std::hash<int32_t>{}(k);
  auto last_d = key_hash % 10;
  bset[last_d] = 1;
}

char BloomFilter::contains_key(int32_t k) {
  auto key_hash = std::hash<int32_t>{}(k);
  auto last_d = key_hash % 10;
  return bset[last_d];
}

void LSM::put(int32_t k, int32_t v, char deleted) {
  if (m.size() >= 4) {
    std::string filename = directory / std::to_string(files.size());
    files.push_back(write_to_file(filename));
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
    for (auto i = files.rbegin(); i != files.rend(); ++i ) {
      File file = *i;
      if (file.bloomFilter.contains_key(k) && k >= file.min && k <= file.max) {
        std::map<int32_t, maybe_value> file_map = read_from_file(file.filename);
        auto it_m = file_map.find(k);
        if (it_m != m.end()) {
          if(!it_m->second.is_deleted) {
            return it_m->second.v;
          } else {
            return std::nullopt;
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

  for (auto file : files) {
    if (r >= file.min && l <= file.max) {
      std::map<int32_t, maybe_value> file_map = read_from_file(file.filename);
      auto it_ml = file_map.lower_bound(l);
      auto it_mu = file_map.lower_bound(r);
      for (auto it=it_ml; it!=it_mu; ++it) {
        if (!it->second.is_deleted) {
          res_map.insert_or_assign(it->first, it->second.v);
        } else {
          res_map.erase(it->first);
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

File LSM::write_to_file(std::string filename) {
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
  return File(filename, bloomFilter, min, max);
}

std::map<int32_t, maybe_value> LSM::read_from_file(std::string filename) {
  std::map<int32_t, maybe_value> new_m;
  if (auto istrm = std::ifstream(filename, std::ios::binary)) {
    int k;
    while (istrm.read(reinterpret_cast<char *>(&k), sizeof k)) {
      int v;
      char d;
      istrm.read(reinterpret_cast<char *>(&v), sizeof v);
      istrm.read(reinterpret_cast<char *>(&d), sizeof d);
      new_m.insert_or_assign(k, maybe_value(v,d));
    }
  } else {
    std::cerr << "could not open " << filename << "\n";
    exit(3);
  }
  return new_m;
}

void LSM::dump_map() {
  for (auto const&x : m) {
    std::cout << x.first << "\n";
  }
}
