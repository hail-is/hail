package is.hail.expr.ir

import is.hail.utils.TreeTraversal

object IRTraversal {

  val postOrder: BaseIR => Iterator[BaseIR] =
    TreeTraversal.postOrder(_.children.iterator)

  val preOrder: BaseIR => Iterator[BaseIR] =
    TreeTraversal.preOrder(_.children.iterator)

  val levelOrder: BaseIR => Iterator[BaseIR] =
    TreeTraversal.levelOrder(_.children.iterator)

  val trace: BaseIR => Iterator[List[BaseIR]] = {
    val adj = { p: List[BaseIR] => p.head.children.map(_ :: p).iterator }
    TreeTraversal.levelOrder(adj) _ compose (_ :: Nil)
  }
}
