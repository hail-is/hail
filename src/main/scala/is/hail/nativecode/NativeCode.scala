package is.hail.nativecode

//
// The mangled name of a companion object contains '$' characters,
// so we can't write C++ native methods directly.
//
object NativeCode {
  val singleton = new NativeCode()

  def getModuleKey(options: String, source: String) =
    singleton.getModuleKey(options, source)

  def findOrBuildModule(options: String, source: String, key: String) =
    singleton.findOrBuildModule(options, source, key)

  def getBuildErrors(key: String) =
    singleton.getBuildErrors(key)

  def readModule(key: String): Array[Byte] =
    singleton.readModule(key)

  def writeModule(key: String, module: Array[Byte]) =
    singleton.writeModule(key, module)
}

class NativeCode() {
  //
  // Generate the "module key" (20 hex digits) corresponding to the
  // compile options and the C++ source text
  //
  @native def getModuleKey(options: String, source: String): String

  //
  // Returns true if the module is ok, false if build fails
  //
  @native def findOrBuildModule(options: String, source: String, key: String): Boolean

  //
  // If findOrBuildModule was false, getBuildErrors(key) gives the build errors
  //
  @native def getBuildErrors(key: String): String

  //
  // We need to replicate the module DLL
  //
  @native def readModule(moduleKey: String): Array[Byte]

  @native def writeModule(moduleKey: String, module: Array[Byte]): Unit
}
