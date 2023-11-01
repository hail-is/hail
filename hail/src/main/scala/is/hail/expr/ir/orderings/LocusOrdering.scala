package is.hail.expr.ir.orderings

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.interfaces.{SLocus, SLocusValue}

object LocusOrdering {
  def make(t1: SLocus, t2: SLocus, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrderingCompareConsistentWithOthers {
      override val type1: SLocus = t1
      override val type2: SLocus = t2

      require(t1.rg == t2.rg)

      override def _compareNonnull(cb: EmitCodeBuilder, lhsc: SValue, rhsc: SValue): Value[Int] = {
        val lhs: SLocusValue = lhsc.asLocus
        val rhs: SLocusValue = rhsc.asLocus

        val ret = cb.newLocal[Int]("locus_cmp_ret", 0)
        def intCmp(l: Code[Int], r: Code[Int]): Code[Int] = {
          Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", l, r)
        }
        cb.assign(ret, intCmp(lhs.contigIdx(cb), rhs.contigIdx(cb)))
        cb.if_(ret.ceq(0), {
          cb.assign(ret, intCmp(lhs.position(cb), rhs.position(cb)))
        })
        ret
      }
    }
  }
}
