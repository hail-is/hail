package is.hail.utils.richUtils

import is.hail.utils.JSONReader
import org.json4s.JValue

class RichJValue(jv: JValue) {
  def fromJSON[T](implicit jr: JSONReader[T]): T = jr.fromJSON(jv)
}
