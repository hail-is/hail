package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.SCanonicalLocusPointer
import is.hail.types.physical.stypes.interfaces.{SLocus, SLocusValue, SStringValue}

object LocusOrdering {
  def make(t1: SLocus, t2: SLocus, ecb: EmitClassBuilder[_]): CodeOrdering = {

    (t1, t2) match {
      case (SCanonicalLocusPointer(_), SCanonicalLocusPointer(_)) =>
        new CodeOrderingCompareConsistentWithOthers {
          val type1: SLocus = t1
          val type2: SLocus = t2

          require(t1.rg == t2.rg)

          def _compareNonnull(cb: EmitCodeBuilder, lhsc: SCode, rhsc: SCode): Code[Int] = {
            val codeRG = cb.emb.getReferenceGenome(t1.rg)
            val lhs: SLocusValue = lhsc.asLocus.memoize(cb, "locus_cmp_lhs")
            val rhs: SLocusValue = rhsc.asLocus.memoize(cb, "locus_cmp_rhs")
            val lhsContig = lhs.contig(cb).memoize(cb, "locus_cmp_lcontig").asInstanceOf[SStringValue]
            val rhsContig = rhs.contig(cb).memoize(cb, "locus_cmp_rcontig").asInstanceOf[SStringValue]

            // ugh
            val lhsContigType = lhsContig.get.st
            val rhsContigType = rhsContig.get.st
            val strcmp = CodeOrdering.makeOrdering(lhsContigType, rhsContigType, ecb)

            val ret = cb.newLocal[Int]("locus_cmp_ret", 0)
            cb.ifx(strcmp.compareNonnull(cb,
              lhsContig.get,
              rhsContig.get).ceq(0), {
              cb.assign(ret, Code.invokeStatic2[java.lang.Integer, Int, Int, Int](
                "compare", lhs.position(cb), rhs.position(cb)))
            }, {
              cb.assign(ret, codeRG.invoke[String, String, Int](
                "compare", lhsContig.get.loadString(), rhsContig.get.loadString()))
            })
            ret
          }
        }
    }
  }

}
