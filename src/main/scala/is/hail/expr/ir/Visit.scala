package is.hail.expr.ir

import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import scala.collection.mutable._

//
// Traversing an IR tree/DAG - if a node has multiple
// parents, it may be visited from each.
//

object Visit {

  def apply(node: BaseIR, visitor: BaseIR => Boolean): Boolean = {
    if (visitor(node))
      true
    else {
      var found = false
      for {
        child <- node.children
        if (!found)
      } found = apply(child, visitor)
      found
    }
  }

}

//
// Traversing an IR tree/DAG, visiting each node only once
//

object VisitOnce {
  
  def apply(root: BaseIR, visitor: BaseIR => Boolean): Boolean = {
    val nodes = new HashSet[BaseIR]()

    def visitDeep(node: BaseIR): Boolean = {
      if (!nodes.add(node)) 
        false
      else {
        var found = false
        for {
          child <- node.children
          if (!found)
        } found = visitDeep(child)
        found
      }
    }

    visitDeep(root)
  }
}

object ContainsAgg {
  def apply(root: BaseIR): Boolean = Visit(root,
    node => node match {
      case ApplyAggOp(_,_,_,_) => true
      case _ => false
    }
  )
}

