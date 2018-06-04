package is.hail.nativecode

//
// NativeLongFunc declares reference-counted handles corresponding to
// C++ functions with various numbers of Long arguments.
//
// The NativeLongFunc also takes a NativeStatus argument so
// that we can pass back an error status and message when appropriate.
//
// Having a separate static type for each function signature
// allows us to get static checking of call signatures (though
// typically a mismatch will result in failing to find the 
// symbol for the mangled function name).
//
// On the C++ side, the function handle also has a keep-alive
// reference to the module containing the function, which must
// not be unloaded while calls are possible.
//

class NativeLongFuncBase() extends NativeBase() {
}

class NativeLongFuncL0 extends NativeLongFuncBase {
  @native def apply(e: NativeStatus): Long
  def copyAssign(b: NativeLongFuncL0) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL0) = super.moveAssign(b)
}

class NativeLongFuncL1() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long): Long
  def copyAssign(b: NativeLongFuncL1) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL1) = super.moveAssign(b)
}

class NativeLongFuncL2() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus, 
                    a0: Long, a1: Long): Long
  def copyAssign(b: NativeLongFuncL2) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL2) = super.moveAssign(b)
}

class NativeLongFuncL3() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long, a1: Long, a2: Long): Long
  def copyAssign(b: NativeLongFuncL3) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL3) = super.moveAssign(b)
}

class NativeLongFuncL4() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long, a1: Long, a2: Long, a3: Long): Long
  def copyAssign(b: NativeLongFuncL4) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL4) = super.moveAssign(b)
}

class NativeLongFuncL5() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long, a1: Long, a2: Long, a3: Long,
                    a4: Long): Long
  def copyAssign(b: NativeLongFuncL5) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL5) = super.moveAssign(b)
}

class NativeLongFuncL6() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long, a1: Long, a2: Long, a3: Long,
                    a4: Long, a5: Long): Long
  def copyAssign(b: NativeLongFuncL6) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL6) = super.moveAssign(b)
}

class NativeLongFuncL7() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long, a1: Long, a2: Long, a3: Long,
                    a4: Long, a5: Long, a6: Long): Long
  def copyAssign(b: NativeLongFuncL7) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL7) = super.moveAssign(b)
}

class NativeLongFuncL8() extends NativeLongFuncBase() {
  @native def apply(e: NativeStatus,
                    a0: Long, a1: Long, a2: Long, a3: Long,
                    a4: Long, a5: Long, a6: Long, a7: Long): Long
  def copyAssign(b: NativeLongFuncL8) = super.copyAssign(b)
  def moveAssign(b: NativeLongFuncL8) = super.moveAssign(b)
}

