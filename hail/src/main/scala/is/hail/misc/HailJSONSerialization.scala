package is.hail.misc

import org.json4s.{Extraction, Formats, JValue, JsonInput}
import org.json4s.jackson.JsonMethods

import java.io.OutputStream

object HailJSONSerialization {
  def write[A <: AnyRef](a: A)(implicit formats: Formats): String =
    JsonMethods.mapper.writeValueAsString(Extraction.decompose(a)(formats).transformField { case ("jsonClass", v) => ("name", v) })

  def write[A <: AnyRef](a: A, out: OutputStream)(implicit formats: Formats): Unit = {
    val transformed = Extraction.decompose(a)(formats: Formats).transformField { case ("jsonClass", v) => ("name", v) }
    JsonMethods.mapper.writeValue(out, transformed)
  }

  def parseSerializedClass(in: JsonInput, useBigDecimalForDouble: Boolean = false, useBigIntForLong: Boolean = true): JValue = {
    JsonMethods.parse(in, useBigDecimalForDouble, useBigIntForLong).transformField { case ("jsonClass", v) => ("name", v) }
  }
}
