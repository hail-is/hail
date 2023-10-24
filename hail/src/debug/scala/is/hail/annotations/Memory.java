package is.hail.annotations;

import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.TreeMap;

@SuppressWarnings("sunapi")
public class Memory {
    private static final Unsafe unsafe;

    private static final TreeMap<Long, Long> blocks;
    private static final Object blocksLock;

    public static void storeByte(byte[] mem, long off, byte b) {
        checkBytes(mem, off, 1);
        unsafe.putByte(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, b);
    }

    public static void storeInt(byte[] mem, long off, int i) {
        checkBytes(mem, off, 4);
        unsafe.putInt(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, i);
    }

    public static void storeLong(byte[] mem, long off, long l) {
        checkBytes(mem, off, 8);
        unsafe.putLong(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, l);
    }

    public static void storeFloat(byte[] mem, long off, float f) {
        checkBytes(mem, off, 4);
        unsafe.putFloat(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, f);
    }

    public static void storeDouble(byte[] mem, long off, double d) {
        checkBytes(mem, off, 8);
        unsafe.putDouble(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, d);
    }

    public static void storeAddress(byte[] mem, long off, long a) {
        checkBytes(mem, off, 8);
        unsafe.putLong(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, a);
    }

    public static byte loadByte(byte[] mem, long off) {
        checkBytes(mem, off, 1);
        return unsafe.getByte(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off);
    }

    public static float loadFloat(byte[] mem, long off) {
        checkBytes(mem, off, 4);
        return unsafe.getFloat(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off);
    }

    public static int loadInt(byte[] mem, long off) {
        checkBytes(mem, off, 4);
        return unsafe.getInt(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off);
    }

    public static long loadLong(byte[] mem, long off) {
        checkBytes(mem, off, 8);
        return unsafe.getLong(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off);
    }

    public static double loadDouble(byte[] mem, long off) {
        checkBytes(mem, off, 8);
        return unsafe.getDouble(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off);
    }

    public static long loadAddress(byte[] mem, long off) {
        checkBytes(mem, off, 8);
        // Unsafe has no getAddress on Object
        return unsafe.getLong(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off);
    }

    public static void memcpy(byte[] dst, long dstOff, byte[] src, long srcOff, long n) {
        if (n > 0) {
            checkBytes(dst, dstOff, n);
            checkBytes(src, srcOff, n);
            unsafe.copyMemory(src, Unsafe.ARRAY_BYTE_BASE_OFFSET + srcOff, dst, Unsafe.ARRAY_BYTE_BASE_OFFSET + dstOff, n);
        }
    }

    // srcOff is in doubles, n is in doubles
    public static void memcpy(byte[] dst, long dstOff, double[] src, long srcOff, long n) {
        if (n > 0) {
            checkBytes(dst, dstOff, n * 8);
            checkDoubles(src, srcOff, n);
            unsafe.copyMemory(src, Unsafe.ARRAY_DOUBLE_BASE_OFFSET + srcOff * 8, dst, Unsafe.ARRAY_BYTE_BASE_OFFSET + dstOff, n * 8);
        }
    }

    // dstOff is in doubles, n is in doubles
    public static void memcpy(double[] dst, long dstOff, byte[] src, long srcOff, long n) {
        if (n > 0) {
            checkDoubles(dst, dstOff, n);
            checkBytes(src, srcOff, n * 8);
            unsafe.copyMemory(src, Unsafe.ARRAY_BYTE_BASE_OFFSET + srcOff, dst, Unsafe.ARRAY_DOUBLE_BASE_OFFSET + dstOff * 8, n * 8);
        }
    }

    public static void memcpy(long dst, byte[] src, long srcOff, long n) {
        if (n > 0) {
            checkAddress(dst, n);
            checkBytes(src, srcOff, n);
            copyFromArray(dst, src, srcOff, n);
        }
    }

    public static void memcpy(byte[] dst, long dstOff, long src, long n) {
        if (n > 0) {
            checkBytes(dst, dstOff, n);
            checkAddress(src, n);
            copyToArray(dst, dstOff, src, n);
        }
    }

    public static void memset(long offset, long size, byte b) {
        checkAddress(offset, size);
        unsafe.setMemory(offset, size, b);
    }

    public static boolean loadBoolean(long addr) {
        checkAddress(addr, 1);
        return unsafe.getByte(addr) != 0;
    }

    public static byte loadByte(long addr) {
        checkAddress(addr, 1);
        return unsafe.getByte(addr);
    }

    public static short loadShort(long addr) {
        checkAddress(addr, 2);
        return unsafe.getShort(addr);
    }

    public static int loadInt(long addr) {
        checkAddress(addr, 4);
        return unsafe.getInt(addr);
    }

    public static long loadLong(long addr) {
        checkAddress(addr, 8);
        return unsafe.getLong(addr);
    }

    public static float loadFloat(long addr) {
        checkAddress(addr, 4);
        return unsafe.getFloat(addr);
    }

    public static double loadDouble(long addr) {
        checkAddress(addr, 8);
        return unsafe.getDouble(addr);
    }

    public static long loadAddress(long addr) {
        checkAddress(addr, 8);
        return unsafe.getAddress(addr);
    }

    public static void storeBoolean(long addr, boolean b) {
        checkAddress(addr, 1);
        unsafe.putByte(addr, (byte)(b ? 1 : 0));
    }

    public static void storeByte(long addr, byte b) {
        checkAddress(addr, 1);
        unsafe.putByte(addr, b);
    }

    public static void storeShort(long addr, short s) {
        checkAddress(addr, 2);
        unsafe.putShort(addr, s);
    }

    public static void storeInt(long addr, int i) {
        checkAddress(addr, 4);
        unsafe.putInt(addr, i);
    }

    public static void storeLong(long addr, long l) {
        checkAddress(addr, 8);
        unsafe.putLong(addr, l);
    }

    public static void storeFloat(long addr, float f) {
        checkAddress(addr, 4);
        unsafe.putFloat(addr, f);
    }

    public static void storeDouble(long addr, double d) {
        checkAddress(addr, 8);
        unsafe.putDouble(addr, d);
    }

    public static void storeAddress(long addr, long a) {
        checkAddress(addr, 8);
        unsafe.putAddress(addr, a);
    }

    static void checkBytes(byte[] mem, long off, long size) {
        if (off < 0 || size < 0 || off + size > mem.length)
            throw new RuntimeException("invalid memory access; off: " + off + " size: " + size);
    }

    static void checkDoubles(double[] mem, long off, long size) {
        if (! (off + size <= mem.length))
            throw new RuntimeException("invalid memory access");
    }

    static void checkAddress(long addr, long size) {
        Map.Entry<Long, Long> e;
        synchronized(blocksLock) {
            e = blocks.floorEntry(addr);
        }
        if (e == null) {
            throw new RuntimeException(String.format("invalid memory access: %08x/%08x: no block", addr, size));
        }

        long blockBase = e.getKey();
        long blockSize = e.getValue();
        if ((addr + size > blockBase + blockSize) || addr < blockBase || size < 0) {
            throw new RuntimeException(String.format("invalid memory access: %08x/%08x: not in %08x/%08x", addr, size, blockBase, blockSize));
        }
    }

    // random
    private static long HEADER = 0x135dc78cc5ea3feeL;
    private static long FOOTER = 0xda52884331476b9L;

    public static long malloc(long size) {
        long addr = unsafe.allocateMemory(size + 16);
        unsafe.putLong(addr, HEADER);
        unsafe.putLong(addr + 8 + size, FOOTER);
        synchronized(blocksLock) {
            blocks.put(addr + 8, size);
        }
        return addr + 8;
    }

    public static void free(long a) {
        Long blockSize;
        synchronized(blocksLock) {
            blockSize = blocks.get(a);
        }
        if (blockSize == null)
            throw new RuntimeException("free invalid memory");
        if (unsafe.getLong(a - 8) != HEADER)
            throw new RuntimeException("corrupt block");
        if (unsafe.getLong(a + blockSize) != FOOTER)
            throw new RuntimeException("corrupt block");

        synchronized(blocksLock) {
            blocks.remove(a);
        }
        unsafe.freeMemory(a - 8);
    }

    public static void memcpy(long dst, long src, long n) {
        if (n > 0) {
            checkAddress(src, n);
            checkAddress(dst, n);
            unsafe.copyMemory(src, dst, n);
        }
    }

    public static void copyToArray(byte[] dst, long dstOff, long src, long n) {
        if (n > 0) {
            checkBytes(dst, dstOff, n);
            checkAddress(src, n);
            unsafe.copyMemory(null, src, dst, Unsafe.ARRAY_BYTE_BASE_OFFSET + dstOff, n);
        }
    }

    public static void copyFromArray(long dst, byte[] src, long srcOff, long n) {
        if (n > 0) {
            checkAddress(dst, n);
            checkBytes(src, srcOff, n);
            unsafe.copyMemory(src, Unsafe.ARRAY_BYTE_BASE_OFFSET + srcOff, null, dst, n);
        }
    }

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

        blocks = new TreeMap<Long, Long>();
        blocksLock = new Object();
    }
}
