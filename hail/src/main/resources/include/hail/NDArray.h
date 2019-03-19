#ifndef HAIL_NDARRAY_H
#define HAIL_NDARRAY_H 1

#include <vector>

struct NDArray {
  int flags; // least sig. bit denotes if row major
  size_t elem_size;
  std::vector<long> shape;
  std::vector<long> strides;
  const char *data;
};

NDArray make_ndarray(int flags, size_t elem_size, std::vector<long> shape, const char *data);
char const *load_ndarray_addr(NDArray nd, std::vector<long> indices);

void set_strides_row_major(std::vector<long> &strides, std::vector<long> &shape);
void set_strides_col_major(std::vector<long> &strides, std::vector<long> &shape);

NDArray make_ndarray(int flags, size_t elem_size, std::vector<long> shape, const char *data) {
  NDArray nd;
  nd.flags = flags;
  nd.elem_size = elem_size;
  nd.shape = shape;
  nd.data = data;

  std::vector<long> strides(shape.size());
  if (flags == 1) {
    set_strides_row_major(strides, shape);
  } else {
    set_strides_col_major(strides, shape);
  }
  nd.strides = strides;

  return nd;
}

char const *load_ndarray_addr(NDArray nd, std::vector<long> indices) {
  if (indices.size() != nd.shape.size()) {
    throw new FatalError("Number of indices must match number of dimensions.");
  }

  int offset = 0;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices[i] < 0 || indices[i] > nd.shape[i]) {
      throw new FatalError("Invalid index");
    }
    offset += nd.strides[i] * indices[i];
  }

  return nd.data + offset * nd.elem_size;
}

void set_strides_row_major(std::vector<long> &strides, std::vector<long> &shape) {
  if (shape.size() > 0) {
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
  }
}

void set_strides_col_major(std::vector<long> &strides, std::vector<long> &shape) {
  if (shape.size() > 0) {
    strides[0] = 1;
    for (int i = 1; i < shape.size(); ++i) {
      strides[i] = shape[i - 1] * strides[i - 1];
    }
  }
}

#endif
