package is.hail.expr.ir

import is.hail.expr._
import is.hail.utils.{ArrayBuilder, _}

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

object ContainsScan {
  def apply(root: BaseIR): Boolean = Exists(root, _.isInstanceOf[ApplyScanOp])
}

object Extract {
  private def extract(node: BaseIR, visitor: BaseIR => Boolean, ab: ArrayBuilder[BaseIR]) {
    if (visitor(node))
      ab += node
    else
      node.children.foreach(extract(_, visitor, ab))
  }

  def apply(node: BaseIR, visitor: BaseIR => Boolean): Array[BaseIR] = {
    val ab = new ArrayBuilder[BaseIR]()
    extract(node, visitor, ab)
    ab.result()
  }
}