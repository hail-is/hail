package is.hail.nativecode

import is.hail.annotations.Memory

//
// NativeStatus holds a C++ std::shared_ptr<NativeStatus> which can be used
// for returning error-status from native C++ functions to Scala.
//
// There's fast direct-field-access from Scala for checking the errCode, but
// more detailed diagnosis needs to go through JNI calls.
//
// We can re-use a NativeStatus across many C++ calls, amortizing the
// cost of creating the object.
//

class NativeStatus() extends NativeBase() {
  @native def nativeCtorErrnoOffset(): Long
  
  val errnoOffset = nativeCtorErrnoOffset()
  
  // Use direct field access to allow fast test for errno != 0
  final def errno: Int = Memory.loadInt(get()+errnoOffset)
  
  final def ok: Boolean = (errno == 0)
  final def fail: Boolean = (errno != 0)
  
  @native def getMsg(): String
  
  @native def getLocation(): String

  def copyAssign(b: NativeStatus) = super.copyAssign(b)
  def moveAssign(b: NativeStatus) = super.moveAssign(b)

  override def toString(): String = {
    if (errno == 0)
      "NoError"
    else
      s"${getLocation()}: ${errno} ${getMsg()}"
  }
}
