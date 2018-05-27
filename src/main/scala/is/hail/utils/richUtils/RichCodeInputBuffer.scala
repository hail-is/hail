package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.io.InputBuffer
import is.hail.utils._
import is.hail.asm4s._

class RichCodeInputBuffer(in: Code[InputBuffer]) {
  def readByte(): Code[Byte] = {
    in.invoke[Byte]("readByte")
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

  def readBytes(toRegion: Code[Region], toOff: Code[Long], n: Int): Code[Unit] = {
    if (n == 0)
      Code._empty
    else if (n < 5)
      Code(
        Code((0 until n).map(i =>
          toRegion.storeByte(toOff + const(i), in.readByte())): _*),
        // for type
        Code._empty)
    else
      in.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)
  }

  def skipBoolean(): Code[Unit] = in.invoke[Unit]("skipBoolean")

  def skipByte(): Code[Unit] = in.invoke[Unit]("skipByte")

  def skipInt(): Code[Unit] = in.invoke[Unit]("skipInt")

  def skipLong(): Code[Unit] = in.invoke[Unit]("skipLong")

  def skipFloat(): Code[Unit] = in.invoke[Unit]("skipFloat")

  def skipDouble(): Code[Unit] = in.invoke[Unit]("skipDouble")

  def skipBytes(n: Code[Int]): Code[Unit] = {
    in.invoke[Int, Unit]("skipBytes", n)
  }
}
