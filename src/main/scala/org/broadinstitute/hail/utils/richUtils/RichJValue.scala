package org.broadinstitute.hail.utils.richUtils

import org.broadinstitute.hail.utils.JSONReader
import org.json4s.JValue

class RichJValue(jv: JValue) {
  def fromJSON[T](implicit jr: JSONReader[T]): T = jr.fromJSON(jv)
}
