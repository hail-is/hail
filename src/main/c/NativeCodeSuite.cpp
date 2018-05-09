// Functions used by NativeCodeSuite.scala

extern "C" {

long hailTestHash1(long a0) {
  return a0 + (a0<<16);
}

long hailTestHash2(long a0, long a1) {
  return a0 + (a1<<4);
}

long hailTestHash3(long a0, long a1, long a2) {
  return a0 + (a1<<4) + (a2<<8);
}

long hailTestHash4(long a0, long a1, long a2, long a3) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12);
}

long hailTestHash5(long a0, long a1, long a2, long a3,
                   long a4) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16);
}

long hailTestHash6(long a0, long a1, long a2, long a3,
                   long a4, long a5) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16) + (a5<<20);
}

long hailTestHash7(long a0, long a1, long a2, long a3,
                   long a4, long a5, long a6) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16) + (a5<<20) + (a6<<24);
}

long hailTestHash8(long a0, long a1, long a2, long a3,
                   long a4, long a5, long a6, long a7) {
  return a0 + (a1<<4) + (a2<<8) + (a3<<12) + (a4<<16) + (a5<<20) + (a6<<24) + (a7<<28);
}

} // end extern "C"
