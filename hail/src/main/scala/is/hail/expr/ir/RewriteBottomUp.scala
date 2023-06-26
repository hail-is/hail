package is.hail.expr.ir

import is.hail.utils.StackSafe._

object RewriteBottomUp {
  def apply(ir: BaseIR, rule: BaseIR => Option[BaseIR]): BaseIR = {
    var rewrite: BaseIR => StackFrame[BaseIR] = null
    rewrite = (ast: BaseIR) =>
      for {
        rewritten <- ast.mapChildrenStackSafe(rewrite)
        result <- rule(rewritten) match {
          case Some(newAST) =>
            if (newAST != rewritten)
              call(rewrite(newAST))
            else done(newAST)
          case None =>
            done(rewritten)
        }
      } yield result

    rewrite(ir).run()
  }
}
