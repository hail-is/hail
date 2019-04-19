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
    long prev_stride = 1;
    int end = shape.size() - 1;
    if (shape[end] == 1) {
      strides[end] = 0;
    } else {
      strides[end] = prev_stride;
    }

    for (int i = shape.size() - 2; i >= 0; --i) {
      if (shape[i] == 1) {
        strides[i] = 0;
      } else {
        strides[i] = shape[i + 1] * prev_stride;
        prev_stride = strides[i];
      }
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
    long prev_stride = 1;
    if (shape[0] == 1) {
      strides[0] = 0;
    } else {
      strides[0] = prev_stride;
    }

    for (int i = 1; i < shape.size(); ++i) {
      if (shape[i] == 1) {
        strides[i] = 0;
      } else {
        strides[i] = shape[i - 1] * prev_stride;
        prev_stride = strides[i];
      }
    }
  }
  return strides;
}

std::vector<long> unify_shapes(std::vector<long> &left, std::vector<long> &right) {
  std::vector<long> result(left.size());

  for (int i = 0; i < left.size(); ++i) {
    if (!(left[i] == right[i] || left[i] == 1 || right[i] == 1)) {
      throw new FatalError("Incompatible shapes for element-wise map");
    }
    result[i] = std::max(left[i], right[i]);
  }

  return result;
}

// Assumption: left and right are of dimension 1 or more
std::vector<long> matmul_shape(std::vector<long> &left, std::vector<long> &right) {
  std::vector<long> result;
  int l_size = left.size();
  int r_size = right.size();

  int left_inner_dim = left[l_size - 1];
  int right_inner_dim = r_size == 1 ? right[0] : right[r_size - 2];
  if (left_inner_dim != right_inner_dim) {
    throw new FatalError("Invalid shapes for matrix multiplication");
  }

  // stack dimensions
  for (int i = 0; i < l_size - 2; i++) {
    if (left[i] != right[i]) {
      throw new FatalError("Invalid shapes for matrix multiplication");
    }
    result.push_back(left[i]);
  }

  // Doesn't work for vectors
  int n = left[l_size - 2];
  result.push_back(n);

  int m = right[r_size - 1];
  result.push_back(m);

  return result;
}
#endif
