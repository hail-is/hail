package is.hail.expr.ir

import is.hail.utils.log

object RewriteBottomUp {
  def apply(ir: BaseIR, rule: (BaseIR) => Option[BaseIR]): BaseIR = {
    var i = 1
    def rewrite(ast: BaseIR): BaseIR = {
      val newChildren = ast.children.map(rewrite)

      // only recons if necessary
      val rewritten =
        if ((ast.children, newChildren).zipped.forall(_ eq _))
          ast
        else {
          val newAST = ast.copy(newChildren)
          log.info(s"before rewrite $i: \n${Pretty(ast)}")
          log.info(s"after rewrite $i: \n${Pretty(newAST)}")
          i += 1
          newAST
        }


      rule(rewritten) match {
        case Some(newAST) if newAST != rewritten =>
          val thing = rewrite(newAST)
          log.info(s"before rewrite $i: \n${Pretty(ast)}")
          log.info(s"after rewrite $i: \n${Pretty(newAST)}")
          i += 1
          thing
        case None =>
          rewritten
      }
    }

    rewrite(ir)
  }
}
