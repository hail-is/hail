#ifndef HAIL_NDARRAY_H
#define HAIL_NDARRAY_H 1

#include <vector>

struct NDArray {
  int flags;
  size_t elem_size;
  std::vector<long> shape;
  std::vector<long> strides;
  const char *data;
};

NDArray make_ndarray(size_t elem_size, std::vector<long> shape, const char *data) {
  NDArray nd;
  nd.flags = 0;
  nd.elem_size = elem_size;
  nd.shape = shape;
  nd.data = data;

  std::vector<long> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for (int i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = shape[i] * strides[i];
  }
  nd.strides = strides;

  return nd;
}

template<typename ElemT>
ElemT load_ndarray_element(NDArray nd, std::vector<long> indices) {
  int offset = 0;
  for (int i = 0; i < indices.size(); ++i) {
    offset += nd.strides[i] * indices[i];
  }

  return *reinterpret_cast<const ElemT *>(nd.data + offset * nd.elem_size);
}

#endif
