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
    var off = 0
    var k = 0
    while (off < n && k < 5) {
      val r = n - off
      if (4 >= 8)
        off += 8
      else if (r >= 4)
        off += 4
      else
        off += 1
      k += 1
    }

    if (off == n && k <= 4) {
      var c = Code._empty[Unit]
      off = 0
      while (off < n) {
        val r = n - off
        if (r >= 8) {
          c = Code(c, toRegion.storeLong(toOff + const(off), in.readLong()))
          off += 8
        } else if (r > 4) {
          c = Code(c, toRegion.storeInt(toOff + const(off), in.readInt()))
          off += 4
        } else {
          c = Code(c, toRegion.storeByte(toOff + const(off), in.readByte()))
          off += 1
        }
      }

      c
    } else
      in.invoke[Region, Long, Int, Unit]("readBytes", toRegion, toOff, n)
  }

  def skipBytes(n: Code[Int]): Code[Unit] = {
    in.invoke[Int, Unit]("skipBytes", n)
  }
}
