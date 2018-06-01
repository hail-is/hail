package is.hail.nativecode

//
// NativePtrFunc declares reference-counted handles corresponding to
// C++ functions which return a std::make_shared<SharedObj>.
//
// These will be called indirectly through the NativePtr.nativePtrFuncN
// methods.

// On the C++ side, the function handle also has a keep-alive
// reference to the module containing the function, which must
// not be unloaded while calls are possible.
//

class NativePtrFuncBase() extends NativeBase() {
}

class NativePtrFuncL0 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL0) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL0) = super.moveAssign(b)
}

class NativePtrFuncL1 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL1) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL1) = super.moveAssign(b)
}

class NativePtrFuncL2 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL2) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL2) = super.moveAssign(b)
}

class NativePtrFuncL3 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL3) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL3) = super.moveAssign(b)
}

class NativePtrFuncL4 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL4) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL4) = super.moveAssign(b)
}

class NativePtrFuncL5 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL5) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL5) = super.moveAssign(b)
}

class NativePtrFuncL6 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL6) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL6) = super.moveAssign(b)
}

class NativePtrFuncL7 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL7) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL7) = super.moveAssign(b)
}

class NativePtrFuncL8 extends NativePtrFuncBase {
  def copyAssign(b: NativePtrFuncL8) = super.copyAssign(b)
  def moveAssign(b: NativePtrFuncL8) = super.moveAssign(b)
}
