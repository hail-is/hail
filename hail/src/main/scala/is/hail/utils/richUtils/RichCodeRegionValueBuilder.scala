package is.hail.utils.richUtils

import is.hail.annotations.{RegionValue, RegionValueBuilder}
import is.hail.asm4s.Code
import is.hail.types.physical.PType

class RichCodeRegionValueBuilder(val rvb: Code[RegionValueBuilder]) {
  def start(pType: Code[PType]): Code[Unit] = {
    rvb.invoke[PType, Unit]("start", pType)
  }

  def startStruct(init: Code[Boolean]): Code[Unit] = {
    rvb.invoke[Boolean, Unit]("startStruct", init)
  }

  def endStruct(): Code[Unit] = {
    rvb.invoke[Unit]("endStruct")
  }

  def startTuple(init: Code[Boolean]): Code[Unit] = {
    rvb.invoke[Boolean, Unit]("startTuple", init)
  }

  def endTuple(): Code[Unit] = {
    rvb.invoke[Unit]("endTuple")
  }

  def end(): Code[Long] = {
    rvb.invoke[Long]("end")
  }
}
