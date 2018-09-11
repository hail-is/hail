package is.hail.expr.ir

import is.hail.utils._

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
          case ApplyUnaryPrimOp(_, _) |
               ApplyBinaryPrimOp(_, _, _) |
               Apply(_, _) |
               ApplySpecial(_, _) |
               Cast(_, _) =>
            Literal(Interpret(ir, optimize = false), ir.typ)

        }
    })
}
