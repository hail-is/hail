package is.hail.expr.ir

import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import scala.collection.mutable._

//
// Search an IR tree for the first node satisfying some condition
//

object Exists {
  def apply(node: BaseIR, visitor: BaseIR => Boolean): Boolean = {
    if (visitor(node))
      true
    else
      node.children.exists(Exists(_, visitor))
  }
}

object ContainsAgg {
  def apply(root: BaseIR): Boolean = Exists(root, _.isInstanceOf[ApplyAggOp])
}
