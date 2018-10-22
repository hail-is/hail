#ifndef HAIL_UTILS_H
#define HAIL_UTILS_H 1

inline char load_byte(char * off) { return *off; }
inline int load_int(char * off) { return *reinterpret_cast<int *>(off); }
inline long load_long(char * off) { return *reinterpret_cast<long *>(off); }
inline float load_float(char * off) { return *reinterpret_cast<float *>(off); }
inline double load_double(char * off) { return *reinterpret_cast<double *>(off); }
inline int load_length(char * off) { return *reinterpret_cast<int *>(off); }
inline char * load_address(char * off) { return reinterpret_cast<char *>(*reinterpret_cast<long *>(off)); }
inline bool load_bit(char * byte_offset, unsigned int bit_offset) {
  return byte_offset[bit_offset >> 3] & (1 << (bit_offset & 0x7));
}

constexpr int n_missing_bytes(int array_len) {
  return ((unsigned long) array_len + 7L) >> 3;
}

inline char * round_up_alignment(char * off, long alignment) {
  return reinterpret_cast<char *>((reinterpret_cast<long>(off) + (alignment - 1)) & ~(alignment - 1));
}


#endif