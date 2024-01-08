package is.hail.expr.ir.orderings

import is.hail.asm4s.Value
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SStringPointer, SStringPointerValue}
import is.hail.types.physical.stypes.interfaces.SString

object StringOrdering {
  def make(t1: SString, t2: SString, ecb: EmitClassBuilder[_]): CodeOrdering = {
    (t1, t2) match {
      case (SStringPointer(_), SStringPointer(_)) =>
        new CodeOrderingCompareConsistentWithOthers {

          override val type1: SString = t1
          override val type2: SString = t2

          override def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
            val bcode1 = x.asInstanceOf[SStringPointerValue]
            val bcode2 = y.asInstanceOf[SStringPointerValue]
            val ord = BinaryOrdering.make(bcode1.binaryRepr.st, bcode2.binaryRepr.st, ecb)
            ord._compareNonnull(cb, bcode1.binaryRepr, bcode2.binaryRepr)
          }
        }

      case (_, _) =>
        new CodeOrderingCompareConsistentWithOthers {

          override val type1: SString = t1
          override val type2: SString = t2

          override def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
            cb.memoize(x.asString.loadString(cb).invoke[String, Int]("compareTo", y.asString.loadString(cb)))
          }
        }
    }
  }
}
