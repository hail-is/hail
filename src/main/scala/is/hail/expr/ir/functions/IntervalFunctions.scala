package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir._
import is.hail.expr.{Apply, ir, types}
import is.hail.expr.types.{TBoolean, TBooleanOptional, TInterval, TStruct}
import is.hail.utils._

object IntervalFunctions extends RegistryFunctions {

  def registerAll(): Unit = {

    registerCode("Interval", Array(tv("T"), tv("T"), TBoolean(), TBoolean()), TInterval(tv("T")), isDet = true) {
      (mb, args) =>
        val srvb = new StagedRegionValueBuilder(mb, TInterval(tv("T")))
        Code(
          srvb.start(),
          srvb.addIRIntermediate(tv("T"))(args(0)),
          srvb.advance(),
          srvb.addIRIntermediate(tv("T"))(args(1)),
          srvb.advance(),
          srvb.addBoolean(args(2).asInstanceOf[Code[Boolean]]),
          srvb.advance(),
          srvb.addBoolean(args(3).asInstanceOf[Code[Boolean]]),
          srvb.advance()
        )
    }

    registerCode("start", TInterval(tv("T")), tv("T")) {
      case (mb, interval: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        region.loadIRIntermediate(tv("T").t)(TInterval(tv("T")).startOffset(interval))
    }

    registerCode("end", TInterval(tv("T")), tv("T")) {
      case (mb, interval: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        region.loadIRIntermediate(tv("T").t)(TInterval(tv("T")).endOffset(interval))
    }

    registerCode("includesStart", TInterval(tv("T")), TBooleanOptional) {
      case (mb, interval: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        TInterval(tv("T")).includeStart(region, interval)
    }

    registerCode("includesEnd", TInterval(tv("T")), TBooleanOptional) {
      case (mb, interval: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        TInterval(tv("T")).includeEnd(region, interval)
    }
  }
}
