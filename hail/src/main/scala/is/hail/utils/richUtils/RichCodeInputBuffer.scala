package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.io.InputBuffer
import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr.types.physical._

class RichCodeInputBuffer(in: Code[InputBuffer]) {
  def readByte(): Code[Byte] = {
    in.invoke[Byte]("readByte")
  }

  def read(buf: Code[Array[Byte]], toOff: Code[Int], n: Code[Int]): Code[Unit] = {
    in.invoke[Array[Byte], Int, Int, Unit]("read", buf, toOff, n)
  }

  def readBoolean(): Code[Boolean] = {
    in.invoke[Boolean]("readBoolean")
  }

  def readInt(): Code[Int] = {
    in.invoke[Int]("readInt")
  }

  def readLong(): Code[Long] = {
    in.invoke[Long]("readLong")
  }

  def readFloat(): Code[Float] = {
    in.invoke[Float]("readFloat")
  }

  def readDouble(): Code[Double] = {
    in.invoke[Double]("readDouble")
  }

  def readBytes(toRegion: Code[Region], toOff: Code[Long], n: Code[Int]): Code[Unit] = {
    in.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)
  }

  def readBytes(toRegion: Value[Region], toOff: Code[Long], n: Int): Code[Unit] = {
    if (n == 0)
      Code._empty
    else if (n < 5)
      Code.memoize(toOff, "ib_ready_bytes_to") { toOff =>
        Code.memoize(in, "ib_ready_bytes_in") { in =>
          Code((0 until n).map(i =>
            Region.storeByte(toOff.get + i.toLong, in.readByte())))
        }
      }
    else
      in.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)
  }

  def readBytesArray(n: Code[Int]): Code[Array[Byte]] = in.invoke[Int, Array[Byte]]("readBytesArray", n)

  def skipBoolean(): Code[Unit] = in.invoke[Unit]("skipBoolean")

  def skipByte(): Code[Unit] = in.invoke[Unit]("skipByte")

  def skipInt(): Code[Unit] = in.invoke[Unit]("skipInt")

  def skipLong(): Code[Unit] = in.invoke[Unit]("skipLong")

  def skipFloat(): Code[Unit] = in.invoke[Unit]("skipFloat")

  def skipDouble(): Code[Unit] = in.invoke[Unit]("skipDouble")

  def skipBytes(n: Code[Int]): Code[Unit] = {
    in.invoke[Int, Unit]("skipBytes", n)
  }

  def readPrimitive(typ: PType): Code[_] = typ.fundamentalType match {
    case _: PBoolean => readBoolean()
    case _: PInt32 => readInt()
    case _: PInt64 => readLong()
    case _: PFloat32 => readFloat()
    case _: PFloat64 => readDouble()
  }
}
