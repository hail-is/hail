package is.hail.expr.ir

import is.hail.utils._

object FoldConstants {
  private[this] def isDeterministic(ir: IR): Boolean = ir match {
    case x: Apply => x.isDeterministic
    case x: ApplySpecial => x.isDeterministic
    case _ => true
  }

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
               Cast(_, _) if isDeterministic(ir) =>
            Literal(Interpret(ir, optimize = false), ir.typ)

        }
    })
}
