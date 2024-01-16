package is.hail.expr.ir


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

  def apply(node: BaseIR, visitor: BaseIR => Boolean): Boolean =
    if (visitor(node))
      true
    else
      node.children.exists(Exists(_, visitor))
}

object Forall {
  def apply(node: IR, visitor: IR => Boolean): Boolean =
    !Exists.inIR(node, n => !visitor(n))
}

object IsScanResult {
  def apply(root: IR): Boolean = root match {
    case _: ApplyScanOp => true
    case AggFold(_, _, _, _, _, isScan) => isScan
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
    case _: AggFold => true
    case AggFold(_, _, _, _, _, isScan) => !isScan
    case AggFilter(_, _, isScan) => !isScan
    case AggExplode(_, _, _, isScan) => !isScan
    case AggGroupBy(_, _, isScan) => !isScan
    case AggArrayPerElement(_, _, _, _, _, isScan) => !isScan
    case _ => false
  }
}

object ContainsAgg {
  def apply(root: IR): Boolean = IsAggResult(root) || (root match {
    case l: AggLet => !l.isScan
    case _: TableAggregate => false
    case _: MatrixAggregate => false
    case _: StreamAgg => false
    case _ => root.children.exists {
        case child: IR => ContainsAgg(child)
        case _ => false
      }
  })
}

object ContainsAggIntermediate {
  def apply(root: IR): Boolean =
    (root match {
      case _: ResultOp => true
      case _: SeqOp => true
      case _: InitOp => true
      case _: CombOp => true
      case _: DeserializeAggs => true
      case _: SerializeAggs => true
      case _: AggStateValue => true
      case _: CombOpValue => true
      case _: InitFromSerializedValue => true
      case _ => false
    }) || root.children.exists {
      case child: IR => ContainsAggIntermediate(child)
      case _ => false
    }
}

object AggIsCommutative {
  def apply(op: AggOp): Boolean = op match {
    case Take() | Collect() | PrevNonnull() | TakeBy(_) | ReservoirSample() | Fold() => false
    case _ => true
  }
}

object ContainsNonCommutativeAgg {
  def apply(root: IR): Boolean = root match {
    case ApplyAggOp(_, _, sig) => !AggIsCommutative(sig.op)
    case _: TableAggregate => false
    case _: MatrixAggregate => false
    case _ => root.children.exists {
        case child: IR => ContainsNonCommutativeAgg(child)
        case _ => false
      }
  }
}

object ContainsScan {
  def apply(root: IR): Boolean = IsScanResult(root) || (root match {
    case l: AggLet => l.isScan
    case _: TableAggregate => false
    case _: MatrixAggregate => false
    case _: StreamAggScan => false
    case _ => root.children.exists {
        case child: IR => ContainsScan(child)
        case _ => false
      }
  })
}
