package is.hail.utils.richUtils

import is.hail.expr._
import is.hail.annotations.Region
import is.hail.asm4s.Code


class RichCodeRegion(val region: Code[Region]) extends AnyVal {
  def size: Code[Long] = region.invoke[Long]("size")

  def copyFrom(other: Code[Region], readStart: Code[Long], writeStart: Code[Long], n: Code[Long]): Code[Unit] = {
    region.invoke[Region, Long, Long, Long, Unit]("copyFrom", other, readStart, writeStart, n)
  }

  def storeInt(off: Code[Long], v: Code[Int]): Code[Unit] = {
    region.invoke[Long,Int,Unit]("storeInt", off, v)
  }

  def storeLong(off: Code[Long], v: Code[Long]): Code[Unit] = {
    region.invoke[Long,Long,Unit]("storeLong", off, v)
  }

  def storeFloat(off: Code[Long], v: Code[Float]): Code[Unit] = {
    region.invoke[Long,Float,Unit]("storeFloat", off, v)
  }

  def storeDouble(off: Code[Long], v: Code[Double]): Code[Unit] = {
    region.invoke[Long,Double,Unit]("storeDouble", off, v)
  }

  def storeAddress(off: Code[Long], a: Code[Long]): Code[Unit] = {
    region.invoke[Long,Long,Unit]("storeAddress", off, a)
  }

  def storeByte(off: Code[Long], b: Code[Byte]): Code[Unit] = {
    region.invoke[Long, Byte, Unit]("storeByte", off, b)
  }

  def storeBytes(off: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] = {
    region.invoke[Long, Array[Byte], Unit]("storeBytes", off, bytes)
  }

  def allocate(alignment: Code[Long], n: Code[Long]): Code[Long] = {
    region.invoke[Long, Long, Long]("allocate", alignment, n)
  }

  def loadBoolean(off: Code[Long]): Code[Boolean] = {
    region.invoke[Long, Boolean]("loadBoolean", off)
  }

  def loadInt(off: Code[Long]): Code[Int] = {
    region.invoke[Long, Int]("loadInt", off)
  }

  def loadLong(off: Code[Long]): Code[Long] = {
    region.invoke[Long, Long]("loadLong", off)
  }

  def loadFloat(off: Code[Long]): Code[Float] = {
    region.invoke[Long, Float]("loadFloat", off)
  }

  def loadDouble(off: Code[Long]): Code[Double] = {
    region.invoke[Long, Double]("loadDouble", off)
  }

  def loadAddress(off: Code[Long]): Code[Long] = {
    region.invoke[Long, Long]("loadAddress", off)
  }

  def loadBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Boolean] = {
    region.invoke[Long, Long, Boolean]("loadBit", byteOff, bitOff)
  }

  def loadPrimitive(typ: Type): Code[Long] => Code[_] = typ.fundamentalType match {
    case _: TBoolean => loadBoolean
    case _: TInt32 => loadInt
    case _: TInt64 => loadLong
    case _: TFloat32 => loadFloat
    case _: TFloat64 => loadDouble
    case _: TArray => loadAddress
    case _: TBinary => loadAddress
    case _: TStruct => off => off
  }

  def appendPrimitive(typ: Type): (Code[_]) => Code[Long] = typ match {
    case _: TBoolean =>
      x => this.appendInt(x.asInstanceOf[Code[Int]])
    case _: TInt32 =>
      x => this.appendInt(x.asInstanceOf[Code[Int]])
    case _: TInt64 =>
      x => this.appendLong(x.asInstanceOf[Code[Long]])
    case _: TFloat32 =>
      x => this.appendFloat(x.asInstanceOf[Code[Float]])
    case _: TFloat64 =>
      x => this.appendDouble(x.asInstanceOf[Code[Double]])
    case _ =>
      throw new UnsupportedOperationException("cannot append non-primitive type: $typ")
  }

  def setBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] = {
    region.invoke[Long, Long, Unit]("setBit", byteOff, bitOff)
  }

  def setBit(byteOff: Code[Long], bitOff: Long): Code[Unit] = {
    region.invoke[Long, Long, Unit]("setBit", byteOff, bitOff)
  }

  def clearBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] = {
    region.invoke[Long, Long, Unit]("clearBit", byteOff, bitOff)
  }

  def storeBit(byteOff: Code[Long], bitOff: Code[Long], b: Code[Boolean]): Code[Unit] = {
    region.invoke[Long, Long, Boolean, Unit]("setBit", byteOff, bitOff, b)
  }

  def appendInt(i: Code[Int]): Code[Long] = {
    region.invoke[Int, Long]("appendInt", i)
  }

  def appendLong(l: Code[Long]): Code[Long] = {
    region.invoke[Long, Long]("appendLong", l)
  }

  def appendFloat(f: Code[Float]): Code[Long] = {
    region.invoke[Float, Long]("appendFloat", f)
  }

  def appendDouble(d: Code[Double]): Code[Long] = {
    region.invoke[Double, Long]("appendDouble", d)
  }

  def appendByte(b: Code[Byte]): Code[Long] = {
    region.invoke[Byte, Long]("appendByte", b)
  }

  def appendBytes(bytes: Code[Array[Byte]]): Code[Long] = {
    region.invoke[Array[Byte], Long]("appendBytes", bytes)
  }

  def appendBytes(bytes: Code[Array[Byte]], bytesOff: Code[Long], n: Code[Int]): Code[Long] = {
    region.invoke[Array[Byte],Long, Int, Long]("appendBytes", bytes, bytesOff, n)
  }

  def appendString(string: Code[String]): Code[Long] = {
    region.invoke[String, Long]("appendString", string)
  }

  def clear(): Code[Unit] = {
    region.invoke[Unit]("clear")
  }
}
