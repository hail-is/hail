package is.hail.expr.ir

object DistinctlyKeyed {
  def apply(node: BaseIR): DistinctKeyedAnalysis = {
    val distinctMemo = Memo.empty[Unit]
    analyze(node, distinctMemo)
    DistinctKeyedAnalysis(distinctMemo)
  }
  private def analyze(node:BaseIR, memo: Memo[Unit]): Unit = {
    def basicChildrenCheck(children: IndexedSeq[BaseIR]): Unit = {
      children.foreach(child => analyze(child, memo))
      if (children.forall(child => memo.contains(child)))
        memo.bind(node, ())
    }
    node match {
      case TableLiteral(_,_,_,_) =>
      case x@TableRead(_, _, _) =>
        if(x.isDistinctlyKeyed)
          memo.bind(node, ())
      case TableParallelize(_,_) =>
      case TableKeyBy(child, keys, _) =>
        analyze(child, memo)
        if (child.typ.key.forall(cKey => keys.contains(cKey)) && memo.contains(child))
          memo.bind(node, ())
      case TableRange(_,_) => memo.bind(node, ())
      case TableFilter(child, _) => basicChildrenCheck(IndexedSeq(child))
      case TableHead(child, _) => basicChildrenCheck(IndexedSeq(child))
      case TableTail(child, _) => basicChildrenCheck(IndexedSeq(child))
      case TableRepartition(child, _, _) => basicChildrenCheck(IndexedSeq(child))
      case TableJoin(left, right, _, _) => basicChildrenCheck(IndexedSeq(left, right))
      case TableIntervalJoin(left, right, _, _) => basicChildrenCheck(IndexedSeq(left, right))
      case TableMultiWayZipJoin(children, _, _) => basicChildrenCheck(children)
      case TableLeftJoinRightDistinct(left, right, _) => basicChildrenCheck(IndexedSeq(left, right))
      case TableMapPartitions(child, _, _, _) => analyze(child, memo)
      case TableMapRows(child, _) => basicChildrenCheck(IndexedSeq(child))
      case TableMapGlobals(child, _) => basicChildrenCheck(IndexedSeq(child))
      case TableExplode(child, _) => analyze(child, memo)
      case TableUnion(children) => children.foreach(child => analyze(child, memo))
      case TableDistinct(child) =>
        memo.bind(node, ())
        analyze(child, memo)
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        memo.bind(node, ())
        analyze(child, memo)
      case TableAggregateByKey(child, _) =>
        memo.bind(node, ())
        analyze(child, memo)
      case TableOrderBy(child, _) => analyze(child, memo)
      case TableRename(child, _, _) => basicChildrenCheck(IndexedSeq(child))
      case TableFilterIntervals(child, _, _) => basicChildrenCheck(IndexedSeq(child))
      case TableToTableApply(child, _) => analyze(child, memo)
      case BlockMatrixToTableApply(_, _, _) =>
      case BlockMatrixToTable(_) =>
      case RelationalLetTable(_, _, body) => basicChildrenCheck(IndexedSeq(body))
      case _: MatrixIR =>
        throw new IllegalArgumentException("MatrixIR should be lowered when it reaches distinct analysis")
      case _: BlockMatrixIR =>
      case ir: IR =>
        ir.children.foreach(child => analyze(child, memo))
    }

  }

}

case class DistinctKeyedAnalysis(distinctMemo: Memo[Unit]) {
  def contains(tableIR: BaseIR): Boolean = {
    distinctMemo.contains(tableIR)
  }
  override def toString: String = distinctMemo.toString
}