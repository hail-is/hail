package is.hail.io

import is.hail.rvd.AbstractRVDSpec

import org.json4s.Extraction
import org.json4s.jackson.JsonMethods

trait Spec extends Serializable {
  override def toString: String = {
    val jv = Extraction.decompose(this)(AbstractRVDSpec.formats)
    JsonMethods.compact(JsonMethods.render(jv))
  }
}
