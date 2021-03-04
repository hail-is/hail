package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.concrete.SCanonicalLocusPointer
import is.hail.types.physical.stypes.interfaces.{SLocus, SStringValue}
import is.hail.types.physical.{PBinary, PCode, PLocusValue}

object LocusOrdering {
  def make(t1: SLocus, t2: SLocus, ecb: EmitClassBuilder[_]): CodeOrdering = {

    (t1, t2) match {
      case (SCanonicalLocusPointer(_), SCanonicalLocusPointer(_)) =>
        new CodeOrderingCompareConsistentWithOthers {
          val type1: SLocus = t1
          val type2: SLocus = t2

          require(t1.rg == t2.rg)

          def _compareNonnull(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Int] = {
            val codeRG = cb.emb.getReferenceGenome(t1.rg)
            val lhs: PLocusValue = lhsc.asLocus.memoize(cb, "locus_cmp_lhs")
            val rhs: PLocusValue = rhsc.asLocus.memoize(cb, "locus_cmp_rhs")
            val lhsContig = lhs.contig(cb).memoize(cb, "locus_cmp_lcontig").asInstanceOf[SStringValue]
            val rhsContig = rhs.contig(cb).memoize(cb, "locus_cmp_rcontig").asInstanceOf[SStringValue]

            // ugh
            val lhsContigBinType = lhsContig.get.asBytes().st
            val rhsContigBinType = rhsContig.get.asBytes().st
            val bincmp = CodeOrdering.makeOrdering(lhsContigBinType, rhsContigBinType, ecb)

            val ret = cb.newLocal[Int]("locus_cmp_ret", 0)
            cb.ifx(bincmp.compareNonnull(cb,
              lhsContig.get.asBytes().asPCode,
              rhsContig.get.asBytes().asPCode).ceq(0), {
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
