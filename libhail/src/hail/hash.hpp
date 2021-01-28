#ifndef HAIL_HASH_HPP_INCLUDED
#define HAIL_HASH_HPP_INCLUDED 1

#include <tuple>

namespace std {

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

}

#endif
