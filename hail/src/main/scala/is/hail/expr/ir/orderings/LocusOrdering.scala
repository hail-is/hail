package is.hail.expr.ir.orderings

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SCanonicalLocusPointer
import is.hail.types.physical.stypes.interfaces.{SLocus, SLocusValue}

object LocusOrdering {
  def make(t1: SLocus, t2: SLocus, ecb: EmitClassBuilder[_]): CodeOrdering = {

    (t1, t2) match {
      case (SCanonicalLocusPointer(_), SCanonicalLocusPointer(_)) =>
        new CodeOrderingCompareConsistentWithOthers {
          override val type1: SLocus = t1
          override val type2: SLocus = t2

          require(t1.rg == t2.rg)

          override def _compareNonnull(cb: EmitCodeBuilder, lhsc: SValue, rhsc: SValue)
            : Value[Int] = {
            val codeRG = cb.emb.getReferenceGenome(t1.rg)
            val lhs: SLocusValue = lhsc.asLocus
            val rhs: SLocusValue = rhsc.asLocus
            val lhsContig = lhs.contig(cb)
            val rhsContig = rhs.contig(cb)

            // ugh
            val lhsContigType = lhsContig.st
            val rhsContigType = rhsContig.st
            val strcmp = CodeOrdering.makeOrdering(lhsContigType, rhsContigType, ecb)

            val ret = cb.newLocal[Int]("locus_cmp_ret", 0)
            cb.if_(
              strcmp.compareNonnull(cb, lhsContig, rhsContig).ceq(0),
              cb.assign(
                ret,
                Code.invokeStatic2[java.lang.Integer, Int, Int, Int](
                  "compare",
                  lhs.position(cb),
                  rhs.position(cb),
                ),
              ),
              cb.assign(
                ret,
                codeRG.invoke[String, String, Int](
                  "compare",
                  lhsContig.loadString(cb).get,
                  rhsContig.loadString(cb).get,
                ),
              ),
            )
            ret
          }
        }
    }
  }

}
