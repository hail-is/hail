package is.hail.utils;

import is.hail.annotations.Region;

public class IntPacker {
    public int shift = 0;
    public int key = 0;
    public int ki = 0;
    public int di = 0;
    public byte[] keys = null;
    public byte[] data = null;

    public void ensureSpace(int keyLen, int dataLen) {
        if (keys == null || keyLen > keys.length) {
            keys = new byte[keyLen];
        }

        if (data == null || dataLen > data.length) {
            data = new byte[dataLen];
        }
    }

    public void resetPack() {
        shift = 0;
        key = 0;
        ki = 0;
        di = 0;
    }

    // assumes that keys and data are properly initialized
    public void resetUnpack() {
        shift = 0;
        ki = 1;
        di = 0;
        if (keys.length > 0) {
            key = (int)keys[0] & 0xFF;
        } else {
            key = 0;
        }
    }

    public void pack(long addr) {
        if (shift == 8) {
            shift = 0;
            keys[ki] = (byte)key;
            ki += 1;
            key = 0;
        }

        int code;
        int v = Region.loadInt(addr);
        if (v > 0 && v < (1 << 8)) {
            code = 0;
        } else if (v > 0 && v < (1 << 16)) {
            code = 1;
        } else if (v > 0 && v < (1 << 24)) {
            code = 2;
        } else {
            code = 3;
        }

        Region.loadBytes(addr, data, di, code + 1);
        key |= code << shift;
        di += code + 1;
        shift += 2;
    }

    public void finish() {
        if (keys.length == 0)
            return;

        keys[ki] = (byte)key;
        ki += 1;
    }

    public void unpack(long addr) {
        if (shift == 8) {
            shift = 0;
            key = ((int)keys[ki]) & 0xFF;
            ki += 1;
        }

        switch ((key >> shift) & 0x3) {
            case 0:
                Region.storeByte(addr, data[di]);
                Region.setMemory(addr + 1, 3L, (byte)0);
                di += 1;
                break;
            case 1:
                Region.storeBytes(addr, data, di, 2);
                Region.setMemory(addr + 2, 2L, (byte)0);
                di += 2;
                break;
            case 2:
                Region.storeBytes(addr, data, di, 3);
                Region.storeByte(addr + 3, (byte)0);
                di += 3;
                break;
            case 3:
                Region.storeBytes(addr, data, di, 4);
                di += 4;
                break;
            default:
                throw new HailException("unreachable code reached, this is a bug");
        }
        shift += 2;
    }
}
