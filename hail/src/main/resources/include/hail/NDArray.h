#ifndef HAIL_NDARRAY_H
#define HAIL_NDARRAY_H 1

#include <vector>
#include <sstream>

struct NDArray {
  int flags; // Not currently used. Will store metadata for numpy compatibility
  int offset;
  size_t elem_size;
  std::vector<long> shape;
  std::vector<long> strides;
  const char *data;
};

NDArray make_ndarray(int flags, int offset, size_t elem_size, std::vector<long> shape, std::vector<long> strides, const char *data);
char const *load_index(NDArray &nd, int index);
int n_elements(std::vector<long> &shape);
std::vector<long> make_strides(int row_major, std::vector<long> &shape);
std::vector<long> strides_row_major(std::vector<long> &shape);
std::vector<long> strides_col_major(std::vector<long> &shape);

NDArray make_ndarray(int flags, int offset, size_t elem_size, std::vector<long> shape, std::vector<long> strides, const char *data) {
  NDArray nd;
  nd.flags = flags;
  nd.offset = offset;
  nd.elem_size = elem_size;
  nd.shape = shape;
  nd.data = data;
  nd.strides = strides;

  return nd;
}

char const *load_index(NDArray &nd, int index) {
  return nd.data + nd.offset + index * nd.elem_size;
}

int n_elements(std::vector<long> &shape) {
  int total = 1;
  for (int i = 0; i < shape.size(); ++i) {
    total *= shape[i];
  }

  return total;
}

std::vector<long> make_strides(int row_major, std::vector<long> &shape) {
  return (row_major == 1) ? strides_row_major(shape) : strides_col_major(shape);
}

std::vector<long> strides_row_major(std::vector<long> &shape) {
  std::vector<long> strides(shape.size());

  if (shape.size() > 0) {
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
  }
  return strides;
}

std::string npy_header(NDArray &nd, const char * numpy_dtype) {
  std::stringstream s;

  s << "{";
  s << "'descr': " << "'" << numpy_dtype << "'" << ", ";
  s << "'fortran_order': False" << ", ";
  s << "'shape': " << "(";
  for (int i = 0; i < nd.shape.size(); ++i) {
    s << nd.shape[i] << ", ";
  }
  s << ")" << "}";

  return s.str();
}

std::vector<long> strides_col_major(std::vector<long> &shape) {
  std::vector<long> strides(shape.size());

  if (shape.size() > 0) {
    strides[0] = 1;
    for (int i = 1; i < shape.size(); ++i) {
      strides[i] = shape[i - 1] * strides[i - 1];
    }
  }
  return strides;
}

#endif
