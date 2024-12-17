package is.hail.expr.ir

import is.hail.utils.TreeTraversal

object IRTraversal {

  def postOrder: BaseIR => Iterator[BaseIR] =
    TreeTraversal.postOrder(_.children.iterator)

  def preOrder: BaseIR => Iterator[BaseIR] =
    TreeTraversal.preOrder(_.children.iterator)

  def levelOrder: BaseIR => Iterator[BaseIR] =
    TreeTraversal.levelOrder(_.children.iterator)

}
