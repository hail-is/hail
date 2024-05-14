package is.hail.asm4s

class HailClassLoader(parent: ClassLoader) extends ClassLoader(parent) {
  def loadOrDefineClass(name: String, b: Array[Byte]): Class[_] = {
    getClassLoadingLock(name).synchronized {
      try
        loadClass(name)
      catch {
        case _: java.lang.ClassNotFoundException =>
          defineClass(name, b, 0, b.length)
      }
    }
  }
}
