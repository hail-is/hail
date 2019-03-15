#ifndef HAIL_NDARRAY_H
#define HAIL_NDARRAY_H 1

#include <vector>

struct NDArray {
  int flags;
  size_t elem_size;
  std::vector<long> shape;
  char *data;
};

NDArray make_ndarray(size_t elem_size, std::vector<long> shape, char *data) {
  NDArray nd;
  nd.flags = 0;
  nd.elem_size = elem_size;
  nd.shape = shape;
  nd.data = data;

  return nd;
}

template<typename ElemT, bool elem_required, size_t elem_size, size_t elem_align>
ElemT load_ndarray_element(NDArray nd, std::vector<long> indices) {
  int offset = 0;
  for (int i = 0; i < indices.size() - 1; ++i) {
    offset += nd.shape[nd.shape.size() - i - 1] * indices[i];
  }
  offset += indices.back();

  return ArrayLoadImpl<ElemT, elem_required, elem_size, elem_align>::load_element(nd.data, offset);
}

#endif
