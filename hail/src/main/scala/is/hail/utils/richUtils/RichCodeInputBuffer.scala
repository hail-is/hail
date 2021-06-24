package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.io.InputBuffer
import is.hail.utils._
import is.hail.asm4s._
import is.hail.types.physical._
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._

class RichCodeInputBuffer(
  val ib: Value[InputBuffer]
) extends AnyVal {
  def close(): Code[Unit] =
    ib.invoke[Unit]("close")

  def seek(offset: Code[Long]): Code[Unit] =
    ib.invoke[Long, Unit]("seek", offset)

  def readByte(): Code[Byte] =
    ib.invoke[Byte]("readByte")

  def read(buf: Code[Array[Byte]], toOff: Code[Int], n: Code[Int]): Code[Unit] =
    ib.invoke[Array[Byte], Int, Int, Unit]("read", buf, toOff, n)

  def readInt(): Code[Int] =
    ib.invoke[Int]("readInt")

  def readLong(): Code[Long] =
    ib.invoke[Long]("readLong")

  def readFloat(): Code[Float] =
    ib.invoke[Float]("readFloat")

  def readDouble(): Code[Double] =
    ib.invoke[Double]("readDouble")

  def readBytes(toRegion: Code[Region], toOff: Code[Long], n: Code[Int]): Code[Unit] =
    ib.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)

  def readBytesArray(n: Code[Int]): Code[Array[Byte]] =
    ib.invoke[Int, Array[Byte]]("readBytesArray", n)

  def skipBoolean(): Code[Unit] =
    ib.invoke[Unit]("skipBoolean")

  def skipByte(): Code[Unit] =
    ib.invoke[Unit]("skipByte")

  def skipInt(): Code[Unit] =
    ib.invoke[Unit]("skipInt")

  def skipLong(): Code[Unit] =
    ib.invoke[Unit]("skipLong")

  def skipFloat(): Code[Unit] =
    ib.invoke[Unit]("skipFloat")

  def skipDouble(): Code[Unit] =
    ib.invoke[Unit]("skipDouble")

  def skipBytes(n: Code[Int]): Code[Unit] =
    ib.invoke[Int, Unit]("skipBytes", n)

  def readDoubles(to: Code[Array[Double]], off: Code[Int], n: Code[Int]): Code[Unit] =
    ib.invoke[Array[Double], Int, Int, Unit]("readDoubles", to, off, n)

  def readDoubles(to: Code[Array[Double]]): Code[Unit] =
    ib.invoke[Array[Double], Unit]("readDoubles", to)

  def readBoolean(): Code[Boolean] =
    ib.invoke[Boolean]("readBoolean")

  def readUTF(): Code[String] =
    ib.invoke[String]("readUTF")

  def readBytes(toRegion: Value[Region], toOff: Code[Long], n: Int): Code[Unit] = {
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

  def readPrimitive(t: Type): SCode = t match {
    case TBoolean => primitive(readBoolean())
    case TInt32 => primitive(readInt())
    case TInt64 => primitive(readLong())
    case TFloat32 => primitive(readFloat())
    case TFloat64 => primitive(readDouble())
  }
}
