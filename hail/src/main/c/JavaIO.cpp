#include "hail/JavaIO.h"

namespace hail {

void write_block_to_buffer(UpcallEnv up, jobject buffer, char * block, int size) {
  auto jbuf = up.getBuffer(size);
  auto byteBuf = up.env()->GetByteArrayElements(jbuf, nullptr);
  memcpy(byteBuf, block, size);
  up.env()->ReleaseByteArrayElements(jbuf, byteBuf, 0);
  up.env()->CallVoidMethod(buffer, up.config()->OutputBlockBuffer_writeBlock_, jbuf, size);
}

void write_bytes_to_block(UpcallEnv up, char * bytes, int n_bytes, char * block, int * block_offset, int block_size, jobject buffer) {
  if (n_bytes + *block_offset < block_size) {
    memcpy(block + *block_offset, bytes, n_bytes);
    *block_offset += n_bytes;
  } else {
    memcpy(block + *block_offset, bytes, block_size - *block_offset);
    write_block_to_buffer(up, buffer, block, block_size);
    memcpy(block, bytes, n_bytes - (block_size - *block_offset));
    *block_offset = n_bytes - (block_size - *block_offset);
  }
}

void write_byte_to_block(UpcallEnv up, char byte, char * block, int * block_offset, int block_size, jobject buffer) {
  write_bytes_to_block(up, &byte, 1, block, block_offset, block_size, buffer);
}

void write_int_to_block(UpcallEnv up, int i, char * block, int * block_offset, int block_size, jobject buffer) {
  write_bytes_to_block(up, reinterpret_cast<char *>(&i), 4, block, block_offset, block_size, buffer);
}

void write_long_to_block(UpcallEnv up, long l, char * block, int * block_offset, int block_size, jobject buffer) {
  write_bytes_to_block(up, reinterpret_cast<char *>(&l), 8, block, block_offset, block_size, buffer);
}

void write_float_to_block(UpcallEnv up, float f, char * block, int * block_offset, int block_size, jobject buffer) {
  write_bytes_to_block(up, reinterpret_cast<char *>(&f), 4, block, block_offset, block_size, buffer);
}

void write_double_to_block(UpcallEnv up, double d, char * block, int * block_offset, int block_size, jobject buffer) {
  write_bytes_to_block(up, reinterpret_cast<char *>(&d), 8, block, block_offset, block_size, buffer);
}
}