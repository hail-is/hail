package is.hail.expr.ir

import is.hail.utils.StackSafe._

object RewriteBottomUp {
  def apply(ir: BaseIR, rule: BaseIR => Option[BaseIR]): BaseIR = {
    var rewrite: BaseIR => StackFrame[BaseIR] = null
    rewrite = (ast: BaseIR) =>
      for {
        newChildren <- call(ast.children.mapRecur(rewrite))
        rewritten =
          if ((ast.children, newChildren).zipped.forall(_ eq _))
            ast
          else
            ast.copy(newChildren)
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
