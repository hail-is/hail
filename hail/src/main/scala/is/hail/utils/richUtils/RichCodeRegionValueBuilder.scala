package is.hail.utils.richUtils

import is.hail.annotations.{RegionValue, RegionValueBuilder}
import is.hail.asm4s.{Code, LineNumber}
import is.hail.types.physical.PType

class RichCodeRegionValueBuilder(val rvb: Code[RegionValueBuilder]) {
  def start(pType: Code[PType])(implicit line: LineNumber): Code[Unit] = {
    rvb.invoke[PType, Unit]("start", pType)
  }

  def startStruct(init: Code[Boolean])(implicit line: LineNumber): Code[Unit] = {
    rvb.invoke[Boolean, Unit]("startStruct", init)
  }

  def endStruct()(implicit line: LineNumber): Code[Unit] = {
    rvb.invoke[Unit]("endStruct")
  }

  def startTuple(init: Code[Boolean])(implicit line: LineNumber): Code[Unit] = {
    rvb.invoke[Boolean, Unit]("startTuple", init)
  }

  def endTuple()(implicit line: LineNumber): Code[Unit] = {
    rvb.invoke[Unit]("endTuple")
  }

  def end()(implicit line: LineNumber): Code[Long] = {
    rvb.invoke[Long]("end")
  }
}
