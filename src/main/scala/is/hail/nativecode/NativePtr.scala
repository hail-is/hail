package is.hail.nativecode

//
// NativePtr is a Scala object which stores a C++ std::shared_ptr<NativeObj>
//
// This allows off-heap objects to be managed with a consistent ref-count
// including references from both Scala and C++.
//
// But to make the reference-counting work correctly and promptly, the 
// Scala code must not allow NativePtr objects to drop into garbage-
// collection.  Use copy-constructor, copyAssign, and moveAssign
// to get this right.
//
// Some care is needed to construct a C++ object with std::make_shared
// and then hold the reference in a Scala NativePtr, so there are
// multiple constructors which indirect through a NativePtrFuncL<N>
// to get that right.
//

class NativePtr() extends NativeBase() {
  //
  // These methods call a C++ function which returns std::shared_ptr<NativeObj>
  // (usually a wrapper around std::make_shared<SubclassOfNativeObj>), and move 
  // the. fields of the shared_ptr into the Scala NativePtr.
  //  
  @native def nativePtrFuncL0(func: Long): Unit
  @native def nativePtrFuncL1(func: Long, a0: Long): Unit
  @native def nativePtrFuncL2(func: Long, a0: Long, a1: Long): Unit
  @native def nativePtrFuncL3(func: Long,
                              a0: Long, a1: Long, a2: Long): Unit
  @native def nativePtrFuncL4(func: Long,
                              a0: Long, a1: Long, a2: Long, a3: Long): Unit
  @native def nativePtrFuncL5(func: Long,
                              a0: Long, a1: Long, a2: Long, a3: Long,
                              a4: Long): Unit
  @native def nativePtrFuncL6(func: Long,
                              a0: Long, a1: Long, a2: Long, a3: Long,
                              a4: Long, a5: Long): Unit
  @native def nativePtrFuncL7(func: Long,
                              a0: Long, a1: Long, a2: Long, a3: Long,
                              a4: Long, a5: Long, a6: Long): Unit
  @native def nativePtrFuncL8(func: Long,
                              a0: Long, a1: Long, a2: Long, a3: Long,
                              a4: Long, a5: Long, a6: Long, a7: Long): Unit
  
  // copy-constructor
  def this(b: NativePtr) {
    this()
    super.copyAssign(b)
  }
  
  def copyAssign(b: NativePtr) = super.copyAssign(b)
  def moveAssign(b: NativePtr) = super.moveAssign(b)
  
  // construct by indirect call to a NativeFunc<N>
  final def this(
    ptrFunc: NativePtrFuncL0
  ) {
    this()
    nativePtrFuncL0(ptrFunc.get());
  }
  
  final def this(
    ptrFunc: NativePtrFuncL1,
    a0: Long
  ) {
    this()
    nativePtrFuncL1(ptrFunc.get(), a0);
  }
  
  final def this(
    ptrFunc: NativePtrFuncL2,
    a0: Long, a1: Long
  ) {
    this()
    nativePtrFuncL2(ptrFunc.get(), a0, a1);
  }
  
  final def this(
    ptrFunc: NativePtrFuncL3,
    a0: Long, a1: Long, a2: Long
  ) {
    this()
    nativePtrFuncL3(ptrFunc.get(), a0, a1, a2);
  }
  
  final def this(
    ptrFunc: NativePtrFuncL4,
    a0: Long, a1: Long, a2: Long, a3: Long
  ) {
    this()
    nativePtrFuncL4(ptrFunc.get(), a0, a1, a2, a3);
  }
      
  final def this(
    ptrFunc: NativePtrFuncL5,
    a0: Long, a1: Long, a2: Long, a3: Long,
    a4: Long
  ) {
    this()
    nativePtrFuncL5(ptrFunc.get(), a0, a1, a2, a3, a4);
  }
      
  final def this(
    ptrFunc: NativePtrFuncL6,
    a0: Long, a1: Long, a2: Long, a3: Long,
    a4: Long, a5: Long
  ) {
    this()
    nativePtrFuncL6(ptrFunc.get(), a0, a1, a2, a3, a4, a5);
  }
      
  final def this(
    ptrFunc: NativePtrFuncL7,
    a0: Long, a1: Long, a2: Long, a3: Long,
    a4: Long, a5: Long, a6: Long
  ) {
    this()
    nativePtrFuncL7(ptrFunc.get(), a0, a1, a2, a3, a4, a5, a6);
  }
      
  final def this(
    ptrFunc: NativePtrFuncL8,
    a0: Long, a1: Long, a2: Long, a3: Long,
    a4: Long, a5: Long, a6: Long, a7: Long
  ) {
    this()
    nativePtrFuncL8(ptrFunc.get(), a0, a1, a2, a3, a4, a5, a6, a7);
  }
      
}
