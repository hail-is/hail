package is.hail.nativecode

class ObjectArray() extends NativeBase() {
  @native def nativeCtorArray(a: Array[Object]): Unit
  @native def nativeCtorO1(a0: Object): Unit
  @native def nativeCtorO2(a0: Object, a1: Object): Unit
  @native def nativeCtorO3(a0: Object, a1: Object, a2: Object): Unit
  @native def nativeCtorO4(a0: Object, a1: Object, a2: Object, a3: Object): Unit
  
  def this(b: ObjectArray) {
    this()
    super.copyAssign(b)
  }
  
  def this(a: Array[Object]) {
    this()
    nativeCtorArray(a)
  }
  
  def this(a0: Object) {
    this()
    nativeCtorO1(a0)
  }

  def this(a0: Object, a1: Object) {
    this()
    nativeCtorO2(a0, a1)
  }

  def this(a0: Object, a1: Object, a2: Object) {
    this()
    nativeCtorO3(a0, a1, a2)
  }

  def this(a0: Object, a1: Object, a2: Object, a3: Object) {
    this()
    nativeCtorO4(a0, a1, a2, a3)
  }
  
  def copyAssign(b: ObjectArray) = super.copyAssign(b)
  def moveAssign(b: ObjectArray) = super.moveAssign(b)
}
