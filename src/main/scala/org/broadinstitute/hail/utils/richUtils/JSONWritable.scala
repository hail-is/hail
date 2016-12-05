package org.broadinstitute.hail.utils.richUtils

import org.broadinstitute.hail.utils.JSONWriter
import org.json4s.JValue

class JSONWritable[T](x: T, jw: JSONWriter[T]) {
  def toJSON: JValue = jw.toJSON(x)
}
