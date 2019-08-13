package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.{Code, coerce}
import is.hail.expr.types.physical._
import is.hail.io.OutputBuffer

class RichCodeOutputBuffer(out: Code[OutputBuffer]) {
  def writeBoolean(b: Code[Boolean]): Code[Unit] = {
    out.invoke[Boolean, Unit]("writeBoolean", b)
  }

  def writeByte(b: Code[Byte]): Code[Unit] = {
    out.invoke[Byte, Unit]("writeByte", b)
  }

  def writeInt(i: Code[Int]): Code[Unit] = {
    out.invoke[Int, Unit]("writeInt", i)
  }

  def writeLong(l: Code[Long]): Code[Unit] = {
    out.invoke[Long, Unit]("writeLong", l)
  }

  def writeFloat(f: Code[Float]): Code[Unit] = {
    out.invoke[Float, Unit]("writeFloat", f)
  }

  def writeDouble(d: Code[Double]): Code[Unit] = {
    out.invoke[Double, Unit]("writeDouble", d)
  }

  def writeBytes(region: Code[Region], off: Code[Long], n: Code[Int]): Code[Unit] = {
    out.invoke[Region, Long, Int, Unit]("writeBytes", region, off, n)
  }

  def writePrimitive(typ: PType): Code[_] => Code[Unit] = typ.fundamentalType match {
    case _: PBoolean => v => writeBoolean(coerce[Boolean](v))
    case _: PInt32 => v => writeInt(coerce[Int](v))
    case _: PInt64 => v => writeLong(coerce[Long](v))
    case _: PFloat32 => v => writeFloat(coerce[Float](v))
    case _: PFloat64 => v => writeDouble(coerce[Double](v))
  }

  def flush(): Code[Unit] = out.invoke[Unit]("flush")
}
