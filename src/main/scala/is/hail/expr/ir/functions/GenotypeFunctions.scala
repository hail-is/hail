package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.ir._
import is.hail.utils._

object GenotypeFunctions {

  def gqFromPL(pl: IR): IR = {
    val body =
      If(ApplyBinaryPrimOp(LT(), Ref("v"), GetTupleElement(Ref("m"), 0)),
        MakeTuple(Seq(Ref("v"), GetTupleElement(Ref("m"), 0))),
        If(ApplyBinaryPrimOp(LT(), Ref("v"), GetTupleElement(Ref("m"), 1)),
          MakeTuple(Seq(GetTupleElement(Ref("m"), 0), Ref("v"))),
          Ref("m")))
    Let("mtup",
      ArrayFold(pl, MakeTuple(Seq(I32(99), I32(99))), "m", "v", body),
      ApplyBinaryPrimOp(Subtract(), GetTupleElement(Ref("mtup"), 1), GetTupleElement(Ref("mtup"), 0)))
  }
}

class GenotypeFunctions {

  def getGQFromPL(t: TArray): (MethodBuilder, Array[Code[_]]) => Code[Int] = {
    case (fb, Array(pl: Code[Long])) =>
      val m = fb.newLocal[Int]
      val m2 = fb.newLocal[Int]
      val i = fb.newLocal[Int]
      val pli = fb.newLocal[Int]
      val len = fb.newLocal[Int]
      Code(
        m := 99,
        m2 := 99,
        i := 0,
        len := t.loadLength(fb.getArg[Region](1), pl),
        Code.whileLoop(i < len,
          pli := fb.getArg[Region](1).load().loadInt(t.loadElement(fb.getArg[Region](1), pl, i)),
          (pli < m).mux(
            Code(m2 := m, m := pli),
            (pli < m2).mux(
              m2 := pli,
              Code._empty
            )
          ),
          i := i + 1
        ),
        m2 - m
      )
  }

  val gqFromPL: IRFunction[Int] = IRFunction[Int]("gqFromPL", TArray(TInt32()), TInt32())(getGQFromPL(TArray(TInt32())))
  val gqFromPL2: IRFunction[Int] = IRFunction[Int]("gqFromPL", TArray(+TInt32()), TInt32())(getGQFromPL(TArray(+TInt32())))
}
