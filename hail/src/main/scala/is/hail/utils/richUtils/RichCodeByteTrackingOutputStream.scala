package is.hail.utils.richUtils

import is.hail.asm4s.Code

class RichCodeByteTrackingOutputStream(val os: Code[ByteTrackingOutputStream]) extends AnyVal {
  def bytesWritten: Code[Long] = os.invoke[Long]("bytesWritten")

  def writeInt(c: Code[Int]): Code[Unit] = os.invoke[Int, Unit]("write", c)

  def write(b: Code[Array[Byte]]): Code[Unit] = os.invoke[Array[Byte], Unit]("write", b)

  def write(b: Code[Array[Byte]], off: Code[Int], len: Code[Int]): Code[Unit] =
    os.invoke[Array[Byte], Int, Int, Unit]("write", b, off, len)

  def close(): Code[Unit] = os.invoke[Unit]("close")


}
