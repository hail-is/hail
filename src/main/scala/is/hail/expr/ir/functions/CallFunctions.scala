package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.variant.Call.alleleRepr
import is.hail.variant._

object CodeAllelePair {
  def apply(j: Code[Int], k: Code[Int]) = new CodeAllelePair(j | (k << 16))
}

class CodeAllelePair(val p: Code[Int]) {
  val j: Code[Int] = p & 0xffff
  val k: Code[Int] = (p >>> 16) & 0xffff
  val nNonRefAlleles: Code[Int] =
    j.ceq(0).mux(0, 1) + k.ceq(0).mux(0, 1)
  val alleleIndices: (Code[Int], Code[Int]) = (j, k)
}

class CallFunctions {
  def diploidGTIndex(j: Code[Int], k: Code[Int]): Code[Int] =
    k * (k + 1) / 2 + j

  def diploidGTIndexWithSwap(i: Code[Int], j: Code[Int]): Code[Int] = {
    (j < i).mux(diploidGTIndex(j, i), diploidGTIndex(i, j))
  }

  def call1(aj: Code[Int], phased: Code[Boolean]): Code[Call] =
    call(aj, phased, 1)

  def call2(aj: Code[Int], ak: Code[Int], phased: Code[Boolean]): Code[Call] =
    call(
      phased.mux(diploidGTIndex(aj, aj + ak),
        diploidGTIndexWithSwap(aj, ak)),
      phased, const(2))

  def call(ar: Code[Int], phased: Code[Boolean], ploidy: Code[Int]): Code[Call] = {
    const(0) | phased.toI | ((ploidy & 3) << 1) | (ar << 3)
  }

  val throwBadPloidy = Code._fatal(s"invalid ploidy. Only support ploidy == 2")

  val ploidy: IRFunction[Int] = IRFunction[Int]("ploidy", TCall(), TInt32()) {
    case (fb, Array(c: Code[Call])) => (c >>> 1) & 0x3
  }

  val isPhased: IRFunction[Boolean] = IRFunction[Boolean]("isPhased", TCall(), TBoolean()) {
    case (fb, Array(call: Code[Call])) => (call & 0x1).ceq(1)
  }

  val isHaploid: IRFunction[Boolean] = IRFunction[Boolean]("isHaploid", TCall(), TBoolean()) {
    case (fb, Array(call: Code[Call])) => ploidy(fb, call).ceq(1)
  }

  val isDiploid: IRFunction[Boolean] = IRFunction[Boolean]("isDiploid", TCall(), TBoolean()) {
    case (fb, Array(call: Code[Call])) => ploidy(fb, call).ceq(2)
  }

  val isUnphasedDiploid: IRFunction[Boolean] = IRFunction[Boolean]("isUnphasedDiploid", TCall(), TBoolean()) {
    case (_, Array(call: Code[Call])) => (call & 0x7).ceq(4)
  }

  val isPhasedDiploid: IRFunction[Boolean] = IRFunction[Boolean]("isPhasedDiploid", TCall(), TBoolean()) {
    case (_, Array(call: Code[Call])) => (call & 0x7).ceq(5)
  }

  val alleleRepr: IRFunction[Int] = IRFunction[Int]("alleleRepr", TCall(), TInt32()) {
    case (_, Array(call: Code[Call])) => call >>> 3
  }

  def allelePair(fb: MethodBuilder, call: Code[Call]): CodeAllelePair = {
    new CodeAllelePair(
      isDiploid(fb, call).mux(
        isPhased(fb, call).mux(
          Code.invokeStatic[AllelePairFunctions, Int, Int]("allelePairFromPhased", alleleRepr(fb, call)),
          Code.invokeStatic[AllelePairFunctions, Int, Int]("allelePair", alleleRepr(fb, call))),
        throwBadPloidy))
  }

  val downcode: IRFunction[Call] = IRFunction[Call]("downcode", TCall(), TInt32(), TCall()) {
    case (fb, Array(call: Code[Call], i: Code[Int])) =>
      val ap = fb.newLocal[Int]("downcode")
      val k = fb.newLocal[Int]("downcode")
      val j = fb.newLocal[Int]("downcode")
      val ip = fb.newLocal[Boolean]("downcode")
      val ploidyvar = fb.newLocal[Int]("downcode_pl")
      Code(
        ploidyvar := ploidy(fb, call),
        ploidyvar.ceq(2).mux(
          Code(
            ap := allelePair(fb, call).p,
            j := new CodeAllelePair(ap).j.ceq(i).mux(1, 0),
            k := new CodeAllelePair(ap).k.ceq(i).mux(1, 0),
            ip := isPhased(fb, call),
            call2(j, k, ip)
          )
          ,
          ploidyvar.ceq(1).mux(
            call1(alleleByIndex(fb, call, 0).ceq(i).toI, isPhased(fb, call)),
            ploidyvar.ceq(0).mux(
              call,
              throwBadPloidy
            )
          )
        )
      )

  }

  def alleleByIndex(fb: MethodBuilder, c: Code[Call], i: Code[Int]): Code[Int] =
    ploidy(fb, c).ceq(1).mux(
      alleleRepr(fb, c),
      ploidy(fb, c).ceq(2).mux(
        i.ceq(0).mux(allelePair(fb, c).j, allelePair(fb, c).k),
        throwBadPloidy
      ))

  val unphasedDiploidGTIndex: IRFunction[Int] = IRFunction[Int]("UnphasedDiploidGtIndexCall", TInt32(), TCall()) {
    case (fb, Array(gt: Code[Int])) =>
      (gt < 0).mux(
        Code._fatal(Code.invokeStatic[java.lang.Integer, Int, String]("toString", gt)),
        call(gt, false, 2)
      )
  }

  val equiv: IRFunction[Boolean] = IRFunction[Boolean]("==", TCall(), TCall(), TBoolean()) {
    case (fb, Array(c1: Code[Call], c2: Code[Call])) =>
      c1.ceq(c2)
  }
}