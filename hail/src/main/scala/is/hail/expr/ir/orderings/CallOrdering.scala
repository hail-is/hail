package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.PCode
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.SCall

object CallOrdering {
  def make(t1: SCall, t2: SCall, ecb: EmitClassBuilder[_]): CodeOrdering = {
    // ugh ugh ugh
    // mistakes were made
    // we made our bed now we lie in it
    new CodeOrderingCompareConsistentWithOthers {
      override val type1: SType = t1
      override val type2: SType = t2

      override def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare",
          x.asCall.loadCanonicalRepresentation(cb), y.asCall.loadCanonicalRepresentation(cb))
      }
    }
  }
}
