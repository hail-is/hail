package is.hail.annotations.ordering;

import is.hail.expr.*;
import is.hail.annotations.*;

public class StaticBinaryUnsafeOrdering {
  public static int compare(MemoryBuffer r1, long o1, MemoryBuffer r2, long o2) {
    int length1 = TBinary.loadLength(r1, o1);
    int length2 = TBinary.loadLength(r2, o2);

    long bOff1 = TBinary.bytesOffset(o1);
    long bOff2 = TBinary.bytesOffset(o2);

    int lim = java.lang.Math.min(length1, length2);

    for (int i = 0; i < lim; ++i) {
      byte b1 = r1.loadByte(bOff1 + i);
      byte b2 = r2.loadByte(bOff2 + i);
      if (b1 != b2)
        return Byte.compare(b1, b2);
    }

    return Integer.compare(length1, length2);
  }
}
