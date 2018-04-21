package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.io.InputBuffer


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

  def skipBytes(n: Code[Int]): Code[Unit] = {
    in.invoke[Int, Unit]("skipBytes", n)
  }
}
