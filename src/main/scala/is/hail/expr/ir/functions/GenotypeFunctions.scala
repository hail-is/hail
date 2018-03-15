package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

object GenotypeFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("gqFromPL", TArray(tv("N", _.isInstanceOf[TInt32])), TInt32()) { (mb, pl: Code[Long]) =>
      val region = mb.getArg[Region](1).load()
      val tPL = TArray(tv("N").t)
      val m = mb.newLocal[Int]("m")
      val m2 = mb.newLocal[Int]("m2")
      val len = mb.newLocal[Int]("len")
      val pli = mb.newLocal[Int]("pli")
      val i = mb.newLocal[Int]("i")
      Code(
        m := 99,
        m2 := 99,
        len := tPL.loadLength(region, pl),
        i := 0,
        Code.whileLoop(i < len,
          tPL.isElementDefined(region, pl, i).mux(
            Code._empty,
            Code._fatal("PL cannot have missing elements.")),
          pli := region.loadInt(tPL.loadElement(region, pl, i)),
          (pli < m).mux(
            Code(m2 := m, m := pli),
            (pli < m2).mux(
              m2 := pli,
              Code._empty)),
          i := i + 1
        ),
        m2 - m
      )
    }
  }
}