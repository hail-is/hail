package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SStringPointer, SStringPointerCode}
import is.hail.types.physical.stypes.interfaces.SString

object StringOrdering {
  def make(t1: SString, t2: SString, ecb: EmitClassBuilder[_]): CodeOrdering = {
    (t1, t2) match {
      case (SStringPointer(_), SStringPointer(_)) =>
        new CodeOrderingCompareConsistentWithOthers {

          val type1: SString = t1
          val type2: SString = t2

          def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] = {
            val bcode1 = x.asInstanceOf[SStringPointerCode]
            val bcode2 = y.asInstanceOf[SStringPointerCode]
            val ord = BinaryOrdering.make(bcode1.binaryRepr.st, bcode2.binaryRepr.st, ecb)
            ord.compareNonnull(cb, bcode1.binaryRepr, bcode2.binaryRepr)
          }
        }

      case (_, _) =>
        new CodeOrderingCompareConsistentWithOthers {

          val type1: SString = t1
          val type2: SString = t2

          def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] = {
            x.asString.loadString().invoke[String, Int]("compareTo", y.asString.loadString())
          }
        }
    }
  }
}
