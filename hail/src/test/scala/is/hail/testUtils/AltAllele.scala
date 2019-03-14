package is.hail.testUtils

import org.apache.spark.sql.Row
import org.json4s._

case class AltAllele(ref: String, alt: String) {
  require(ref != alt, "ref was equal to alt")
  require(!ref.isEmpty, "ref was an empty string")
  require(!alt.isEmpty, "alt was an empty string")

  override def toString: String = s"$ref/$alt"
}
