package is.hail.expr.ir

import is.hail.expr._
import is.hail.utils.{ArrayBuilder, _}

import scala.collection.mutable._

//
// Search an IR tree for the first node satisfying some condition
//

object Exists {
  def inIR(node: IR, visitor: IR => Boolean): Boolean = {
    if (visitor(node))
      true
    else
      node.children.exists {
        case child: TableAggregate => visitor(child)
        case child: MatrixAggregate => visitor(child)
        case child: IR => inIR(child, visitor)
        case _ => false
      }
  }

  def apply(node: BaseIR, visitor: BaseIR => Boolean): Boolean = {
    if (visitor(node))
      true
    else
      node.children.exists(Exists(_, visitor))
  }
}

object Forall {
  def apply(node: IR, visitor: IR => Boolean): Boolean =
    !Exists.inIR(node, n => !visitor(n))
}

object IsScanResult {
  def apply(root: IR): Boolean = root match {
    case _: ApplyScanOp => true
    case AggFilter(_, _, isScan) => isScan
    case AggExplode(_, _, _, isScan) => isScan
    case AggGroupBy(_, _, isScan) => isScan
    case AggArrayPerElement(_, _, _, _, _, isScan) => isScan
    case _ => false
  }
}

object IsAggResult {
  def apply(root: IR): Boolean = root match {
    case _: ApplyAggOp => true
    case AggFilter(_, _, isScan) => !isScan
    case AggExplode(_, _, _, isScan) => !isScan
    case AggGroupBy(_, _, isScan) => !isScan
    case AggArrayPerElement(_, _, _, _, _, isScan) => !isScan
    case _ => false
  }
}

object ContainsAgg {
  def apply(root: IR): Boolean = IsAggResult(root) || (root match {
    case _: TableAggregate => false
    case _: MatrixAggregate => false
    case ArrayAgg(a, _, _) => ContainsAgg(a)
    case _ => root.children.exists {
      case child: IR => ContainsAgg(child)
      case _ => false
    }
  })
}

object ContainsScan {
  def apply(root: IR): Boolean = IsScanResult(root) || (root match {
    case _: TableAggregate => false
    case _: MatrixAggregate => false
    case ArrayAggScan(a, _, _) => ContainsScan(a)
    case _ => root.children.exists {
      case child: IR => ContainsScan(child)
      case _ => false
    }
  })
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