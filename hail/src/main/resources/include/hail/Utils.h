#ifndef HAIL_UTILS_H
#define HAIL_UTILS_H 1

char load_byte(char * off) { return *off; }
int load_int(char * off) { return *reinterpret_cast<int *>(off); }
long load_long(char * off) { return *reinterpret_cast<long *>(off); }
float load_float(char * off) { return *reinterpret_cast<float *>(off); }
double load_double(char * off) { return *reinterpret_cast<double *>(off); }
int load_length(char * off) { return *reinterpret_cast<int *>(off); }
char * load_address(char * off) { return reinterpret_cast<char *>(*reinterpret_cast<long *>(off)); }
bool load_bit(char * byte_offset, unsigned int bit_offset) {
  return byte_offset[bit_offset >> 3] & (1 << (bit_offset & 0x7));
}

int n_missing_bytes(int array_len) {
  return ((unsigned long) array_len + 7L) >> 3;
}

char * round_up_alignment(char * off, long alignment) {
  return reinterpret_cast<char *>((reinterpret_cast<long>(off) + (alignment - 1)) & ~(alignment - 1));
}


#endif