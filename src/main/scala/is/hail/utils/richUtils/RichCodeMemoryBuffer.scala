package is.hail.utils.richUtils

import is.hail.expr._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s.Code


class RichCodeMemoryBuffer(val region: Code[MemoryBuffer]) extends AnyVal {

  def size: Code[Long] = region.invoke[Long]("size")

  def offset: Code[Long] = region.invoke[Long]("size")

  def copyFrom(other: Code[MemoryBuffer], readStart: Code[Long], writeStart: Code[Long], n: Code[Long]): Code[Unit] = {
    region.invoke[MemoryBuffer, Long, Long, Long, Unit]("copyFrom", other, readStart, writeStart, n)
  }

  def storeInt32(off: Code[Long], v: Code[Int]): Code[Unit] = {
    region.invoke[Long,Int,Unit]("storeInt", off, v)
  }

  def storeInt64(off: Code[Long], v: Code[Long]): Code[Unit] = {
    region.invoke[Long,Long,Unit]("storeLong", off, v)
  }

  def storeFloat32(off: Code[Long], v: Code[Float]): Code[Unit] = {
    region.invoke[Long,Float,Unit]("storeFloat", off, v)
  }

  def storeFloat64(off: Code[Long], v: Code[Double]): Code[Unit] = {
    region.invoke[Long,Double,Unit]("storeDouble", off, v)
  }

  def storeAddress(off: Code[Long], a: Code[Long]): Code[Unit] = {
    region.invoke[Long,Long,Unit]("storeAddress", off, a)
  }

  def storeByte(off: Code[Long], b: Code[Byte]): Code[Unit] = {
    region.invoke[Long, Byte, Unit]("storeByte", off, b)
  }

  def align(alignment: Code[Long]): Code[Unit] = {
    region.invoke[Long, Unit]("align", alignment)
  }

  def allocate(n: Code[Long]): Code[Long] = {
    region.invoke[Long, Long]("allocate",n)
  }

  def alignAndAllocate(n: Code[Long]): Code[Long] = {
    region.invoke[Long, Long]("alignAndAllocate",n)
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

  def loadPrimitive(typ: Type): Code[Long] => Code[_] = typ match {
    case _: TBoolean =>
      this.loadBoolean(_)
    case _: TInt32 =>
      this.loadInt(_)
    case _: TInt64 =>
      this.loadLong(_)
    case _: TFloat32 =>
      this.loadFloat(_)
    case _: TFloat64 =>
      this.loadDouble(_)
    case _ =>
      off => off
  }

  def appendPrimitive(typ: Type): (Code[_]) => Code[Long] = typ match {
    case _: TBoolean =>
      x => this.appendInt32(x.asInstanceOf[Code[Int]])
    case _: TInt32 =>
      x => this.appendInt32(x.asInstanceOf[Code[Int]])
    case _: TInt64 =>
      x => this.appendInt64(x.asInstanceOf[Code[Long]])
    case _: TFloat32 =>
      x => this.appendFloat32(x.asInstanceOf[Code[Float]])
    case _: TFloat64 =>
      x => this.appendFloat64(x.asInstanceOf[Code[Double]])
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

  def appendInt32(v: Code[Int]): Code[Long] = {
    region.invoke[Int,Long]("appendInt", v)
  }

  def appendInt64(v: Code[Long]): Code[Long] = {
    region.invoke[Long,Long]("appendLong", v)
  }

  def appendFloat32(v: Code[Float]): Code[Long] = {
    region.invoke[Float,Long]("appendFloat", v)
  }

  def appendFloat64(v: Code[Double]): Code[Long] = {
    region.invoke[Double,Long]("appendDouble", v)
  }

  def appendAddress(a: Code[Long]): Code[Long] = {
    region.invoke[Long,Long]("appendAddress", a)
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

  def clear(): Code[Unit] = {
    region.invoke[Unit]("clear")
  }

}
