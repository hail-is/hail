#ifndef HAIL_JAVAIO_H
#define HAIL_JAVAIO_H 1
#include <jni.h>
#include "hail/Upcalls.h"

namespace hail {

void write_block_to_buffer(UpcallEnv env, jobject buffer, char * block, int size);

void write_bytes_to_block(UpcallEnv up, char * bytes, int n_bytes, char * block, int * block_offset, int block_size, jobject buffer);

void write_byte_to_block(UpcallEnv up, char byte, char * block, int * block_offset, int block_size, jobject buffer);

void write_int_to_block(UpcallEnv up, int i, char * block, int * block_offset, int block_size, jobject buffer);
void write_long_to_block(UpcallEnv up, long l, char * block, int * block_offset, int block_size, jobject buffer);
void write_float_to_block(UpcallEnv up, float f, char * block, int * block_offset, int block_size, jobject buffer);
void write_double_to_block(UpcallEnv up, double d, char * block, int * block_offset, int block_size, jobject buffer);

void write_packed_int_to_block(UpcallEnv up, int i, char * block, int * block_offset, int block_size, jobject buffer);

}

#endif