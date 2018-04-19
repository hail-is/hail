package is.hail.expr.ir

import is.hail.utils._
import is.hail.expr.BaseIR

object FoldConstants {
  def apply(ir: BaseIR): BaseIR =
    RewriteBottomUp(ir, matchErrorToNone[BaseIR, BaseIR] {
      case ir: IR =>
        if (!IsScalarType(ir.typ) ||
          !ir.children.forall {
            case c: IR => IsScalarConstant(c)
            case _ => false
          })
          throw new MatchError(ir)

        ir match {
          case If(NA(_), _, _, _) =>
            NA(ir.typ)

          case ApplyUnaryPrimOp(_, _, _) |
               ApplyBinaryPrimOp(_, _, _, _) |
               Apply(_, _, _) |
               ApplySpecial(_, _, _) |
               Cast(_, _) =>
            Literal(Interpret(ir, optimize = false), ir.typ)
        }
    })
}
