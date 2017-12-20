package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.expr.{TArray, TBinary, TBoolean, TFloat32, TFloat64, TInt32, TInt64, TStruct, Type}
import is.hail.io.Decoder

class RichCodeDecoder(val dec: Code[Decoder]) extends AnyVal {
  def readByte(): Code[Byte] = dec.invoke[Byte]("readByte")

  def readInt(): Code[Int] = dec.invoke[Int]("readInt")

  def readLong(): Code[Long] = dec.invoke[Long]("readLong")

  def readFloat(): Code[Float] = dec.invoke[Float]("readFloat")

  def readDouble(): Code[Double] = dec.invoke[Double]("readDouble")

  def readBytes(region: Code[Region], toOff: Code[Long], n: Code[Int]): Code[Unit] =
    dec.invoke[Region,Long,Int, Unit]("readBytes", region, toOff, n)

  def readBinary(region: Code[Region]): Code[Long] =
    dec.invoke[Region, Long]("readBinary", region)

  def readPrimitive(t: Type): Code[_] = t match {
    case _: TBoolean => readByte()
    case _: TInt32 => readInt()
    case _: TInt64 => readLong()
    case _: TFloat32 => readFloat()
    case _: TFloat64 => readDouble()
  }
}
