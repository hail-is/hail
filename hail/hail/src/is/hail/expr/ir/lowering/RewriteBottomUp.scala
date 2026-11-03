package is.hail.expr.ir.lowering

import is.hail.expr.ir.BaseIR
import is.hail.utils.StackSafe._

object RewriteBottomUp {
  def apply(ir: BaseIR, rule: BaseIR => Option[BaseIR]): BaseIR = {
    var rewrite: BaseIR => StackFrame[BaseIR] = null
    rewrite = (ast: BaseIR) =>
      ast.mapChildrenStackSafe(rewrite).flatMap { rewritten =>
        rule(rewritten) match {
          case Some(newAST) if newAST != rewritten => call(rewrite(newAST))
          case Some(newAST) => done(newAST)
          case None => done(rewritten)
        }
      }

    rewrite(ir).run()
  }
}
