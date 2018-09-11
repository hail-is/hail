package is.hail.variant

import org.json4s._

case class Sample(id: String) {
  override def toString: String = id

  def toJSON: JValue = {
    JObject(("id", JString(id)))
  }
}
