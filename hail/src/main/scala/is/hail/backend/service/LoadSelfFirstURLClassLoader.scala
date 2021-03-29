package is.hail.backend.service

import java.net._
import org.apache.log4j.Logger

class LoadSelfFirstURLClassLoader(
  urls: Array[URL]
) extends URLClassLoader(urls) {
  private[this] val log = Logger.getLogger(getClass.getName())

  override protected def loadClass(name: String, resolve: Boolean): Class[_] = synchronized {
    val klass = findLoadedClass(name)
    if (klass != null) {
      return klass
    }
    try {
      val klass = findClass(name)
      if (resolve)
        resolveClass(klass)
      klass
    } catch {
      case e: ClassNotFoundException =>
        super.loadClass(name, resolve)
    }
  }

  override def getResource(name: String): URL = {
    val url = findResource(name)
    if (url != null) {
      return url
    }
    return super.getResource(name)
  }

  private[this] class PrependElementEnumeration[T](
    private[this] val t: T,
    private[this] val rest: java.util.Enumeration[T]
  ) extends java.util.Enumeration[T] {
    private[this] var consumed = false
    def hasMoreElements() = !consumed || rest.hasMoreElements
    def nextElement(): T = {
      if (!consumed) {
        consumed = true
        t
      } else {
        rest.nextElement()
      }
    }
  }

  override def getResources(name: String): java.util.Enumeration[URL] = {
    val url = findResource(name)
    val rest = super.getResources(name)
    if (url != null) {
      return new PrependElementEnumeration(url, rest)
    }
    return rest
  }
}
