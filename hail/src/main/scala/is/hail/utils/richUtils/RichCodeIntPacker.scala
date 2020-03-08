package is.hail.utils.richUtils

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.utils.IntPacker

class RichCodeIntPacker(val p: Code[IntPacker]) extends AnyVal {
  def ki: Code[Int] = p.getField[Int]("ki")
  def di: Code[Int] = p.getField[Int]("di")
  def keys: Code[Array[Byte]] = p.getField[Array[Byte]]("keys")
  def data: Code[Array[Byte]] = p.getField[Array[Byte]]("data")

  def ensureSpace(keyLen: Code[Int], dataLen: Code[Int]): Code[Unit] = p.invoke[Int, Int, Unit]("ensureSpace", keyLen, dataLen)
  def resetPack(): Code[Unit] = p.invoke[Unit]("resetPack")
  def resetUnpack(): Code[Unit] = p.invoke[Unit]("resetUnpack")

  def pack(addr: Code[Long]): Code[Unit] = p.invoke[Long, Unit]("pack", addr)
  def finish(): Code[Unit] = p.invoke[Unit]("finish")
  def unpack(addr: Code[Long]): Code[Unit] = p.invoke[Long, Unit]("unpack", addr)
}
