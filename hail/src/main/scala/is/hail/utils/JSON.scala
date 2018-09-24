package is.hail.utils

import org.json4s.JValue

trait JSONWriter[T] {
  def toJSON(x: T): JValue
}

trait JSONReader[T] {
  def fromJSON(jv: JValue): T
}

trait JSONReaderWriter[T] extends JSONReader[T] with JSONWriter[T]
