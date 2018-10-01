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

void write_packed_int_to_block(UpcallEnv up, int i, char * block, int * block_offset, int block_size, jobject buffer) {
  unsigned int foo = i;
  do {
    unsigned char b = foo & 0x7f;
    foo >>= 7;
    if (foo != 0) {
      b |= 0x80;
    }
    write_byte_to_block(up, static_cast<char>(b), block, block_offset, block_size, buffer);
  } while (foo != 0);
}

}