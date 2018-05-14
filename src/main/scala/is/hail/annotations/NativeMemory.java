//
// NativeByteArray.java - transitional support for expandable Region on C++ heap
//
// Richard Cownie, Hail Team, 2018-03-29
//
/*
package is.hail.nativecall;

import sun.misc.Unsafe;
import java.lang.AutoCloseable;
import java.lang.reflectField;

//
// The NativeByteArray class can be used to replace Region's "mem: Array[Byte]" with
// minimal disruption to all the other code related to Region and RegionValue.
//
// It holds a "long" referring to a malloc()'ed buffer, which can't be moved 
// around by JVM garbage-collection and thus can be accessed from C++ code.
//
// In addition to keeping the buffer off the JVM heap, we also need to explicitly
// free the buffers when they're no longer needed, rather than waiting for
// garbage collection.
//

public class NativeByteArray implements AutoCloseable {
  private static final Unsafe unsafe;
  private static final long kOffLength = 0x0;
  private static final long kOffData   = 0x8;

  //
  // The first 8 bytes of the buffer will be 32bit capacity and size
  //  
  private long buf = 0;
  
  private long getCapacity() {
    return ((buf == 0) ? 0 : unsafe.loadInt(buf+kOffCapacity));
  }
  
  private long getSize() {
    return ((buf == 0) ? 0 : unsafe.loadInt(buf+kOffSize));
  }
  
  private void setSize(long newSize) {
    unsafe.storeInt(newSize);
  }
  
  //
  // The close() method may be called by AutoCloseable, so that we don't
  // lose the malloc'ed memory resource.
  //  
  public void close() {
    if (buf != 0) {
      unsafe.freeMemory(buf);
      buf = 0;
    }
    capacity = 0;
    size = 0;
  }
  
  public void clear() {
    close();
  }
  
  //
  // Store methods
  //
  public void storeByte(long off, byte b) { unsafe.storeByte(buf+off, b); }
  public void storeShort(long off, short b) { unsafe.storeShort(buf+off, b); }
  public void storeInt(long off, int b) { unsafe.storeInt(buf+off, b); }
  public void storeLong(long off, long b) { unsafe.storeLong(buf+off, b); }
  public void storeFloat(long off, float b) { unsafe.storeFloat(buf+off, b); }
  public void storeDouble(long off, double b) { unsafe.storeDouble(buf+off, b); }
  
  //
  // Load methods
  //
  public byte loadByte(long off) { return unsafe.loadByte(off); }
  public short loadShort(long off) { return unsafe.loadShort(off); }
  public int loadInt(long off) { return unsafe.loadInt(off); }
  public long loadLong(long off) { return unsafe.loadLong(off); }
  public float loadFloat(long off) { return unsafe.loadFloat(off); }
  public double loadDouble(long off) { return unsafe.loadDouble(off); }
    
  // Boilerplate code to gain access to sun.misc.Unsafe load/store functions 
  static {
    Unsafe t;
    try {
      Field unsafeField = Unsafe.class.getDeclaredField("theUnsafe");
      unsafeField.setAccessible(true);
      t = (sun.misc.Unsafe) unsafeField.get(null);
    } catch (Throwable cause) {
      t = null;
    }
    unsafe = t;
  }
  
}

//
// A small wrapper giving the minimal interface needed from annotations.Memory
//

public final class Memory {

  public static void storeByte(NativeByteArray mem, long off, byte b) {
    mem.storeByte(off, b);
  }
  public static void storeShort(NativeByteArray mem, long off, short b) {
    mem.storeShort(off, b);
  }
  public static void storeInt(NativeByteArray mem, long off, int b) {
    mem.storeInt(off, b);
  }
  public static void storeLong(NativeByteArray mem, long off, long b) {
    mem.storeLong(off, b);
  }
  public static void storeFloat(NativeByteArray mem, long off, float b) {
    mem.storeFloat(off, b);
  }
  public static void storeDouble(NativeByteArray mem, long off, double b) {
    mem.storeDouble(off, b);
  }
  public static void storeAddress(NativeByteArray mem, long off, long b) {
    mem.storeLong(off, b);
  }
  public static void storeBoolean(NativeByteArray mem, long off, boolean b) {
    mem.storeByte(off, b ? 1 : 0);
  }
  
  public static byte loadByte(NativeByteArray mem, long off) {
    return mem.loadByte(off);
  }
  public static short loadShort(NativeByteArray mem, long off) {
    return mem.loadShort(off);
  }
  public static int loadInt(NativeByteArray mem, long off) {
    return mem.loadInt(off);
  }
  public static long loadLong(NativeByteArray mem, long off) {
    return mem.loadLong(off);
  }
  public static float loadFloat(NativeByteArray mem, long off) {
    return mem.loadFloat(off);
  }
  public static double loadDouble(NativeByteArray mem, long off) {
    return mem.loadDouble(off);
  }
  public static long loadAddress(NativeByteArray mem, long off) {
    return mem.loadAddress(off);
  }
  public static boolean loadBoolean(NativeByteArray mem, long off) {
    return (mem.loadByte(off) != 0);
  }

}
*/