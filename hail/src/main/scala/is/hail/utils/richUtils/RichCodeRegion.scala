package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual._

class RichCodeRegion(val region: Code[Region]) extends AnyVal {
  def allocate(alignment: Code[Long], n: Code[Long]): Code[Long] = {
    region.invoke[Long, Long, Long]("allocate", alignment, n)
  }

  def loadBoolean(off: Code[Long]): Code[Boolean] = {
    region.invoke[Long, Boolean]("loadBoolean", off)
  }

  def loadByte(off: Code[Long]): Code[Byte] = {
    region.invoke[Long, Byte]("loadByte", off)
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

  def loadBytes(off: Code[Long], n: Code[Int]): Code[Array[Byte]] = {
    region.invoke[Long, Int, Array[Byte]]("loadBytes", off, n)
  }

  def loadIRIntermediate(typ: PType): Code[Long] => Code[_] = loadIRIntermediate(typ.virtualType)

  def loadIRIntermediate(typ: Type): Code[Long] => Code[_] = typ.fundamentalType match {
    case _: TBoolean => loadBoolean
    case _: TInt32 => loadInt
    case _: TInt64 => loadLong
    case _: TFloat32 => loadFloat
    case _: TFloat64 => loadDouble
    case _: TArray => loadAddress
    case _: TBinary => loadAddress
    case _: TBaseStruct => off => off
  }

  def getIRIntermediate(typ: PType): Code[Long] => Code[_] = getIRIntermediate(typ.virtualType)

  def getIRIntermediate(typ: Type): Code[Long] => Code[_] = typ.fundamentalType match {
    case _: TBoolean => loadBoolean
    case _: TInt32 => loadInt
    case _: TInt64 => loadLong
    case _: TFloat32 => loadFloat
    case _: TFloat64 => loadDouble
    case _ => off => off
  }

  def clear(): Code[Unit] = { region.invoke[Unit]("clear") }

  def reference(other: Code[Region]): Code[Unit] =
    region.invoke[Region, Unit]("reference", other)

  def setNumParents(n: Code[Int]): Code[Unit] =
    region.invoke[Int, Unit]("setNumParents", n)

  def setParentReference(r: Code[Region], i: Code[Int]): Code[Unit] =
    region.invoke[Region, Int, Unit]("setParentReference", r, i)

  def getParentReference(r: Code[Region], i: Code[Int], size: Int): Code[Region] =
    region.invoke[Int, Int, Region]("getParentReference", i, size)

  def setFromParentReference(r: Code[Region], i: Code[Int], size: Int): Code[Unit] =
    region.invoke[Region, Int, Int, Unit]("setFromParentReference", r, i, size)

  def unreferenceRegionAtIndex(i: Code[Int]): Code[Unit] =
    region.invoke[Int, Unit]("unreferenceRegionAtIndex", i)

  def isValid: Code[Boolean] = region.invoke[Boolean]("isValid")

  def invalidate(): Code[Unit] = region.invoke[Unit]("invalidate")

  def getNewRegion(blockSize: Code[Int]): Code[Unit] = region.invoke[Int, Unit]("getNewRegion", blockSize)
}
