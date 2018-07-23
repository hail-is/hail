#include <cstdint>
#include "hail/NativeStatus.h"

// Functions used by NativeCodeSuite.scala

extern "C" {

int64_t hailTestHash1(NativeStatus*, int64_t a0) {
  return a0 + (a0<<16);
}

int64_t hailTestHash2(NativeStatus*, int64_t a0, int64_t a1) {
  return a0 + (a1<<4);
}

int64_t hailTestHash3(NativeStatus*, int64_t a0, int64_t a1, int64_t a2) {
  return a0 + (a1<<4) + (a2<<8);
}

int64_t hailTestHash4(NativeStatus*, int64_t a0, int64_t a1, int64_t a2, int64_t a3) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12);
}

int64_t hailTestHash5(NativeStatus*, int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                   int64_t a4) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16);
}

int64_t hailTestHash6(NativeStatus*, int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                   int64_t a4, int64_t a5) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16) + (a5<<20);
}

int64_t hailTestHash7(NativeStatus*, int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                   int64_t a4, int64_t a5, int64_t a6) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16) + (a5<<20) + (a6<<24);
}

int64_t hailTestHash8(NativeStatus*, int64_t a0, int64_t a1, int64_t a2, int64_t a3,
                   int64_t a4, int64_t a5, int64_t a6, int64_t a7) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16) + (a5<<20) + (a6<<24) + (a7<<28);
}

} // end extern "C"
