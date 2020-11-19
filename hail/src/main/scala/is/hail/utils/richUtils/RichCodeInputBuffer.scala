package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.io.InputBuffer
import is.hail.utils._
import is.hail.asm4s._
import is.hail.types.physical._

class RichCodeInputBuffer(
  val ib: Value[InputBuffer]
) extends AnyVal {
  def close()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("close")

  def seek(offset: Code[Long])(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Long, Unit]("seek", offset)

  def readByte()(implicit line: LineNumber): Code[Byte] =
    ib.invoke[Byte]("readByte")

  def read(buf: Code[Array[Byte]], toOff: Code[Int], n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Array[Byte], Int, Int, Unit]("read", buf, toOff, n)

  def readInt()(implicit line: LineNumber): Code[Int] =
    ib.invoke[Int]("readInt")

  def readLong()(implicit line: LineNumber): Code[Long] =
    ib.invoke[Long]("readLong")

  def readFloat()(implicit line: LineNumber): Code[Float] =
    ib.invoke[Float]("readFloat")

  def readDouble()(implicit line: LineNumber): Code[Double] =
    ib.invoke[Double]("readDouble")

  def readBytes(toRegion: Code[Region], toOff: Code[Long], n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)

  def readBytesArray(n: Code[Int])(implicit line: LineNumber): Code[Array[Byte]] =
    ib.invoke[Int, Array[Byte]]("readBytesArray", n)

  def skipBoolean()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("skipBoolean")

  def skipByte()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("skipByte")

  def skipInt()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("skipInt")

  def skipLong()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("skipLong")

  def skipFloat()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("skipFloat")

  def skipDouble()(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Unit]("skipDouble")

  def skipBytes(n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Int, Unit]("skipBytes", n)

  def readDoubles(to: Code[Array[Double]], off: Code[Int], n: Code[Int])(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Array[Double], Int, Int, Unit]("readDoubles", to, off, n)

  def readDoubles(to: Code[Array[Double]])(implicit line: LineNumber): Code[Unit] =
    ib.invoke[Array[Double], Unit]("readDoubles", to)

  def readBoolean()(implicit line: LineNumber): Code[Boolean] =
    ib.invoke[Boolean]("readBoolean")

  def readUTF()(implicit line: LineNumber): Code[String] =
    ib.invoke[String]("readUTF")

  def readBytes(toRegion: Value[Region], toOff: Code[Long], n: Int)(implicit line: LineNumber): Code[Unit] = {
    if (n == 0)
      Code._empty
    else if (n < 5)
      Code.memoize(toOff, "ib_ready_bytes_to") { toOff =>
        Code.memoize(ib, "ib_ready_bytes_in") { ib =>
          Code((0 until n).map(i =>
            Region.storeByte(toOff.get + i.toLong, ib.readByte())))
        }
      }
    else
      ib.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)
  }

  def readPrimitive(typ: PType)(implicit line: LineNumber): Code[_] = typ.fundamentalType match {
    case _: PBoolean => readBoolean()
    case _: PInt32 => readInt()
    case _: PInt64 => readLong()
    case _: PFloat32 => readFloat()
    case _: PFloat64 => readDouble()
  }
}
