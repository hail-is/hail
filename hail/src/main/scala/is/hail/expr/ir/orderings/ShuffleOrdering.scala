package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SCanonicalShufflePointer, SCanonicalShufflePointerCode}
import is.hail.types.physical.stypes.interfaces.SShuffle

object ShuffleOrdering {
  def make(t1: SShuffle, t2: SShuffle, ecb: EmitClassBuilder[_]): CodeOrdering = {
    (t1, t2) match {
      case (SCanonicalShufflePointer(_), SCanonicalShufflePointer(_)) =>
        new CodeOrderingCompareConsistentWithOthers {

          val type1: SShuffle = t1
          val type2: SShuffle = t2

          def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] = {
            val bcode1 = x.asInstanceOf[SCanonicalShufflePointerCode].binaryRepr
            val bcode2 = y.asInstanceOf[SCanonicalShufflePointerCode].binaryRepr
            val ord = BinaryOrdering.make(bcode1.st, bcode2.st, ecb)
            ord.compareNonnull(cb, bcode1, bcode2)
          }
        }
    }
  }
}
