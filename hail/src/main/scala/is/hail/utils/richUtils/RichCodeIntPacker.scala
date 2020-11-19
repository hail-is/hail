package is.hail.utils.richUtils

import is.hail.asm4s.{Code, LineNumber}
import is.hail.utils.IntPacker

class RichCodeIntPacker(val p: Code[IntPacker]) extends AnyVal {
  def ki(implicit line: LineNumber): Code[Int] =
    p.getField[Int]("ki")
  def di(implicit line: LineNumber): Code[Int] =
    p.getField[Int]("di")
  def keys(implicit line: LineNumber): Code[Array[Byte]] =
    p.getField[Array[Byte]]("keys")
  def data(implicit line: LineNumber): Code[Array[Byte]] =
    p.getField[Array[Byte]]("data")

  def ensureSpace(keyLen: Code[Int], dataLen: Code[Int])(implicit line: LineNumber): Code[Unit] =
    p.invoke[Int, Int, Unit]("ensureSpace", keyLen, dataLen)
  def resetPack()(implicit line: LineNumber): Code[Unit] =
    p.invoke[Unit]("resetPack")
  def resetUnpack()(implicit line: LineNumber): Code[Unit] =
    p.invoke[Unit]("resetUnpack")

  def pack(addr: Code[Long])(implicit line: LineNumber): Code[Unit] =
    p.invoke[Long, Unit]("pack", addr)
  def finish()(implicit line: LineNumber): Code[Unit] =
    p.invoke[Unit]("finish")
  def unpack(addr: Code[Long])(implicit line: LineNumber): Code[Unit] =
    p.invoke[Long, Unit]("unpack", addr)
}
