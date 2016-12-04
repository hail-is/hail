package org.broadinstitute.hail.utils

import org.json4s.JValue

trait JSONWriter[T] {
  def toJSON(x: T): JValue
}

trait JSONReader[T] {
  def fromJSON(jv: JValue): T
}

trait JSONReaderWriter[T] extends JSONReader[T] with JSONWriter[T]

class RichJSONWritable[T](x: T, jw: JSONWriter[T]) {
  def toJSON: JValue = jw.toJSON(x)
}

class RichJValue(jv: JValue) {
  def fromJSON[T](implicit jr: JSONReader[T]): T = jr.fromJSON(jv)
}
