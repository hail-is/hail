package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
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
}
