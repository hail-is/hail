#ifndef HAIL_UTILS_H
#define HAIL_UTILS_H 1

inline char load_byte(char * off) { return *off; }
inline bool load_bool(char * off) { return *off; }
inline int load_int(char * off) { return *reinterpret_cast<int *>(off); }
inline long load_long(char * off) { return *reinterpret_cast<long *>(off); }
inline float load_float(char * off) { return *reinterpret_cast<float *>(off); }
inline double load_double(char * off) { return *reinterpret_cast<double *>(off); }
inline int load_length(char * off) { return *reinterpret_cast<int *>(off); }
inline char * load_address(char * off) { return reinterpret_cast<char *>(*reinterpret_cast<long *>(off)); }
inline bool load_bit(char * byte_offset, unsigned int bit_offset) {
  return byte_offset[bit_offset >> 3] & (1 << (bit_offset & 0x7));
}


inline void store_byte(char * off, char b) { *off = b; }
inline void store_bool(char * off, bool b) { *off = b; }
inline void store_int(char * off, int i) { *reinterpret_cast<int *>(off) = i; }
inline void store_long(char * off, long l) { *reinterpret_cast<long *>(off) = l; }
inline void store_float(char * off, float f) { *reinterpret_cast<float *>(off) = f; }
inline void store_double(char * off, double d) { *reinterpret_cast<double *>(off) = d; }
inline void store_length(char * off, int len) { *reinterpret_cast<int *>(off) = len; }
inline void store_address(char * off, char * addr) { *reinterpret_cast<long *>(off) = reinterpret_cast<long>(addr); }
inline void store_bit(char * byte_offset, unsigned int bit_offset, bool b) {
  auto off = byte_offset + (bit_offset >> 3);
  auto byte = *off;
  auto new_byte = b ? (byte | (1 << (bit_offset & 0x7))) : (byte & ~(1 << (bit_offset & 0x7)));
  *off = new_byte;
}

constexpr int n_missing_bytes(int array_len) {
  return ((unsigned long) array_len + 7L) >> 3;
}

inline char * round_up_alignment(char * off, long alignment) {
  return reinterpret_cast<char *>((reinterpret_cast<long>(off) + (alignment - 1)) & ~(alignment - 1));
}

inline long round_up_alignment(long off, long alignment) {
  return (off + (alignment - 1)) & ~(alignment - 1);
}

inline int floordiv(int n, int d) {
  int q = n / d;
  int r = n - q * d;
  if (r < 0)
    --q;
  return q;
}

inline long lfloordiv(long n, long d) {
  long q = n / d;
  long r = n - q * d;
  if (r < 0)
    --q;
  return q;
}

#endif
