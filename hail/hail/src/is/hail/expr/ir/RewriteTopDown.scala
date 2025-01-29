package is.hail.expr.ir

object RewriteTopDown {
  def rewriteTopDown(ast: BaseIR, rule: PartialFunction[BaseIR, BaseIR]): BaseIR = {
    def rewrite(ast: BaseIR): BaseIR = {
      rule.lift(ast) match {
        case Some(newAST) if newAST != ast =>
          rewrite(newAST)
        case None =>
          ast.mapChildren(rewrite)
      }
    }

    rewrite(ast)
  }
}
