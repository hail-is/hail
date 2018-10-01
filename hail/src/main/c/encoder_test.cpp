#include "hail/encoder_test.h"
#include "hail/ObjectArray.h"
#include "hail/NativeObj.h"
#include "hail/Upcalls.h"
#include "hail/JavaIO.h"
#include <jni.h>

namespace hail {

// TStruct(TInt32(), TString())
long write_rows(jobject jbuffer, jobject rows) {
  UpcallEnv up;
  JNIEnv * env = up.env();

  int block_size = 32 * 1024;
  char * block_buffer = new char[block_size]{};
  int off = 0;

  while (env->CallBooleanMethod(rows, up.config()->Iterator_hasNext_)) {
    char * row = reinterpret_cast<char *>(env->CallLongMethod(rows, up.config()->Iterator_next_));

    write_byte_to_block(up, 1, block_buffer, &off, block_size, jbuffer);
    write_bytes_to_block(up, row, 1, block_buffer, &off, block_size, jbuffer);
    if (!(row[0] & 0x1)) {
      write_packed_int_to_block(up, *reinterpret_cast<int *>(row + 4), block_buffer, &off, block_size, jbuffer);
    }
    if (!(row[0] & 0x2)) {
      auto off2 = reinterpret_cast<char *>(*reinterpret_cast<long *>(row + 8));
      int length = *reinterpret_cast<int *>(off2);
      write_packed_int_to_block(up, length, block_buffer, &off, block_size, jbuffer);
      write_bytes_to_block(up, off2 + 4, length, block_buffer, &off, block_size, jbuffer);
    }
    if (!(row[0] & 0x4)) {
      auto off2 = reinterpret_cast<char *>(*reinterpret_cast<long *>(row + 16));
      int length = *reinterpret_cast<int *>(off2);
      write_packed_int_to_block(up, length, block_buffer, &off, block_size, jbuffer);
      int nMissingBytes = (length + 7) >> 3;
      char * elements = off2 + ( (4 + nMissingBytes + 7) >> 3 << 3);
      write_bytes_to_block(up, off2 + 4, nMissingBytes, block_buffer, &off, block_size, jbuffer);
      for(int i = 0; i < length; i++) {
        if ((off2[4 + (i >> 3)] & (1 << (i & 0x7))) == 0) {
          write_bytes_to_block(up, elements + (8 * i), 8, block_buffer, &off, block_size, jbuffer);
        }
      }
    }
  }
  write_byte_to_block(up, 0, block_buffer, &off, block_size, jbuffer);
  write_block_to_buffer(up, jbuffer, block_buffer, off);
  return 0;
}

// StreamBlockOutputBuffer - length + block (byte array)
// LZ4OutputBlockBuffer - uncompressed length + LZ4-compressed block (blockSize = 32 * 1024)
// BlockingBufferSpec - write things into blocks until block is full, then write block (blockSize = 32 * 1024)
// LEB128BufferSpec - pack ints and longs using continuation bit
// PackCodecSpec - inline all pointed-at values

}

