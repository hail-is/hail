package is.hail.expr.ir

import is.hail.utils.StackSafe._

object RewriteBottomUp {
  def areObjectEqual(oldChildren: IndexedSeq[BaseIR], newChildren: IndexedSeq[BaseIR]): Boolean = {
    var same = true
    var i = 0
    while (same && i < oldChildren.length) {
      same = oldChildren(i) eq newChildren(i)
      i += 1
    }
    same
  }
  def apply(ir: BaseIR, rule: BaseIR => Option[BaseIR]): BaseIR = {
    var rewrite: BaseIR => StackFrame[BaseIR] = null
    rewrite = (ast: BaseIR) =>
      for {
        newChildren <- call(ast.children.mapRecur(rewrite))
        rewritten = {
          if (areObjectEqual(ast.children, newChildren))
            ast
          else
            ast.copy(newChildren)
        }
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
