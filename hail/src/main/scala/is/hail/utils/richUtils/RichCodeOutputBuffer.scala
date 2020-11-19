package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.physical._
import is.hail.io.OutputBuffer

class RichCodeOutputBuffer(
  val ob: Value[OutputBuffer]
) extends AnyVal {
  def flush()(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Unit]("flush")

  def close()(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Unit]("close")

  def indexOffset()(implicit line: LineNumber): Code[Long] =
    ob.invoke[Long]("indexOffset")

  def writeByte(b: Code[Byte])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Byte, Unit]("writeByte", b)

  def write(buf: Code[Array[Byte]])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Array[Byte], Unit]("write", buf)

  def write(buf: Code[Array[Byte]], startPos: Code[Int], endPos: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Array[Byte], Int, Int, Unit]("write", buf, startPos, endPos)

  def writeInt(i: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Int, Unit]("writeInt", i)

  def writeLong(l: Code[Long])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Long, Unit]("writeLong", l)

  def writeFloat(f: Code[Float])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Float, Unit]("writeFloat", f)

  def writeDouble(d: Code[Double])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Double, Unit]("writeDouble", d)

  def writeBytes(region: Code[Region], off: Code[Long], n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Region, Long, Int, Unit]("writeBytes", region, off, n)

  def writeBytes(addr: Code[Long], n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Long, Int, Unit]("writeBytes", addr, n)

  def writeDoubles(from: Code[Array[Double]], fromOff: Code[Int], n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Array[Double], Int, Int, Unit]("writeDoubles", from, fromOff, n)

  def writeDoubles(from: Code[Array[Double]])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Array[Double], Unit]("writeDoubles", from)

  def writeBoolean(b: Code[Boolean])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[Boolean, Unit]("writeBoolean", b)

  def writeUTF(s: Code[String])(implicit line: LineNumber): Code[Unit] =
    ob.invoke[String, Unit]("writeUTF", s)

  def writePrimitive(typ: PType)(implicit line: LineNumber): Code[_] => Code[Unit] = typ.fundamentalType match {
    case _: PBoolean => v => writeBoolean(coerce[Boolean](v))
    case _: PInt32 => v => writeInt(coerce[Int](v))
    case _: PInt64 => v => writeLong(coerce[Long](v))
    case _: PFloat32 => v => writeFloat(coerce[Float](v))
    case _: PFloat64 => v => writeDouble(coerce[Double](v))
  }
}
