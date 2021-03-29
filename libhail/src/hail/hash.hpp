#ifndef HAIL_HASH_HPP_INCLUDED
#define HAIL_HASH_HPP_INCLUDED 1

#include <tuple>

namespace std {

template<typename T>
size_t hash_value(const T &v) {
  return std::hash<T>{}(v);
}

template<typename T> void
hash_combine(size_t &seed, const T &v) {
  seed ^= hash_value(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename... T>
struct hash<tuple<T...>> {
  size_t operator()(const tuple<T...> &t) const;
};

template<typename... T, size_t... I> size_t
hash_tuple_impl(const tuple<T...> &t, std::index_sequence<I...>) {
  size_t h = sizeof...(T);
  (hash_combine(h, get<I>(t)), ...);
  return h;
}

template<typename... T> size_t
hash<tuple<T...>>::operator()(const tuple<T...> &t) const {
  return hash_tuple_impl(t, std::make_index_sequence<sizeof...(T)>{});
}


template<typename T>
struct hash<vector<T>> {
  size_t operator()(const vector<T> &v) const;
};

template<typename T> size_t
hash<vector<T>>::operator()(const vector<T> &v) const {
  size_t h = v.size();
  for (const auto &x : v)
    hash_combine(h, x);
  return h;
}

}

#endif
