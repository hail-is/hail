package is.hail.nativecode

import com.sun.jna.Native

//
// NativeBase is a Scala object which stores a C++ std::shared_ptr<NativeObj>
//
// This allows off-heap objects to be managed with a consistent ref-count
// including references from both Scala and C++.
//
// But to make the reference-counting work correctly and promptly, the
// Scala code must not allow NativeBase objects to drop into garbage-
// collection.  Use copy-constructor, copyAssign, and moveAssign
// to get this right.
//

class NativeBase() extends AutoCloseable {
  protected var addrA: Long = 0
  protected var addrB: Long = 0

  // This forces the libhail to be loaded with RTLD_GLOBAL
  NativeCode.forceLoad()
  
  // Native methods
  @native def nativeCopyCtor(b_addrA: Long, b_addrB: Long): Unit

  @native def nativeReset(a_addrA: Long, a_addrB: Long): Unit

  @native def nativeUseCount(a_addrA: Long, a_addrB: Long): Long

  // These are protected so that subclasses can enforce type safety
  @native protected def copyAssign(b: NativeBase): Unit
  
  @native protected def moveAssign(b: NativeBase): Unit
  
  // copy-constructor
  final def this(b: NativeBase) {
    this()
    nativeCopyCtor(b.addrA, b.addrB)
  }

  final def close() {
    if (addrA != 0) {
      val tmpA = addrA
      val tmpB = addrB
      addrA = 0
      addrB = 0
      nativeReset(tmpA, tmpB);
    }
  }
  
  final def get(): Long = addrA
  
  final def reset(): Unit = close()
  
  final def use_count(): Long = nativeUseCount(addrA, addrB)

}
