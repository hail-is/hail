#ifndef HAIL_NDARRAY_H
#define HAIL_NDARRAY_H 1

#include <vector>

struct NDArray {
  int flags;
  size_t elem_size;
  std::vector<int> shape;
  std::vector<int> stride;
  char *data;
};

NDArray make_ndarray(size_t elem_size, std::vector<int> shape, char *data);

template<typename T>
T load_ndarray_element(NDArray nd, std::vector<int> indices);

NDArray make_ndarray(size_t elem_size, std::vector<int> shape, char *data) {
  NDArray nd;
  nd.flags = 0;
  nd.elem_size = elem_size;
  nd.shape = shape;
  nd.data = data;

  return nd;
}

template<typename T>
T load_ndarray_element(NDArray nd, std::vector<int> indices) {
  int offset = 0;
  for (int i = 0; i < nd.shape.size() - 1; i++) {
    offset += nd.shape[nd.shape.size() - i - 1] * indices[i] * nd.elem_size;
  }
  offset += indices.back() * nd.elem_size;
  return *reinterpret_cast<T *>(nd.data + offset);
}

#endif
