package is.hail.utils.richUtils

import is.hail.annotations.MemoryBuffer
import is.hail.asm4s.Code


class RichCodeMemoryBuffer(val region: Code[MemoryBuffer]) extends AnyVal {

  def size: Code[Long] = region.invoke[Long]("size")

  def offset: Code[Long] = region.invoke[Long]("size")

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

  def storeBytes(off: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] = {
    region.invoke[Long, Array[Byte], Unit]("storeBytes", off, bytes)
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

  def appendInt32(i: Code[Int]): Code[Unit] = {
    region.invoke[Int, Unit]("appendInt", i)
  }

  def appendInt64(l: Code[Long]): Code[Unit] = {
    region.invoke[Long, Unit]("appendLong", l)
  }

  def appendFloat32(f: Code[Float]): Code[Unit] = {
    region.invoke[Float, Unit]("appendFloat", f)
  }

  def appendFloat64(d: Code[Double]): Code[Unit] = {
    region.invoke[Double, Unit]("appendDouble", d)
  }

  def appendByte(b: Code[Byte]): Code[Unit] = {
    region.invoke[Byte, Unit]("appendByte", b)
  }

  def appendBytes(bytes: Code[Array[Byte]]): Code[Unit] = {
    region.invoke[Array[Byte], Unit]("appendBytes", bytes)
  }

  def appendBytes(bytes: Code[Array[Byte]], bytesOff: Code[Long], n: Code[Int]): Code[Unit] = {
    region.invoke[Array[Byte],Long, Int, Unit]("appendBytes", bytes, bytesOff, n)
  }

  def clear(): Code[Unit] = {
    region.invoke[Unit]("clear")
  }

}
