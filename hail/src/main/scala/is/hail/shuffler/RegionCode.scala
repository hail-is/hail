package is.hail.shuffler

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.types.physical._

object RegionCode {
  def pretty(pType: PType, off: Code[Long]): Code[String] =
    Code.invokeScalaObject2[PType, Long, String](
      Region.getClass, "pretty", Wire.deserializePType(const(Wire.serializePType(pType))), off)
}
