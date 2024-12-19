package is.hail.expr.ir

object DistinctlyKeyed {

  def apply(node: BaseIR): DistinctKeyedAnalysis = {
    val memo = Memo.empty[Unit]
    IRTraversal.postOrder(node).foreach {
      case t: TableRead =>
        memo.bindIf(t.isDistinctlyKeyed, t, ())

      case t @ TableKeyBy(child, keys, _) =>
        memo.bindIf(child.typ.key.forall(keys.contains) && memo.contains(child), t, ())

      case t @ (_: TableFilter
          | _: TableIntervalJoin
          | _: TableLeftJoinRightDistinct
          | _: TableMapRows
          | _: TableMapGlobals) =>
        memo.bindIf(memo.contains(t.children.head), t, ())

      case t @ RelationalLetTable(_, _, body) =>
        memo.bindIf(memo.contains(body), t, ())

      case t @ (_: TableHead
          | _: TableTail
          | _: TableRepartition
          | _: TableJoin
          | _: TableMultiWayZipJoin
          | _: TableRename
          | _: TableFilterIntervals) =>
        memo.bindIf(t.children.forall(memo.contains), t, ())

      case t @ (_: TableRange
          | _: TableDistinct
          | _: TableKeyByAndAggregate
          | _: TableAggregateByKey) =>
        memo.bind(t, ())

      case _: MatrixIR =>
        throw new IllegalArgumentException(
          "MatrixIR should be lowered when it reaches distinct analysis"
        )

      case _ =>
        memo
    }
    DistinctKeyedAnalysis(memo)
  }
}

case class DistinctKeyedAnalysis(distinctMemo: Memo[Unit]) {
  def contains(tableIR: BaseIR): Boolean =
    distinctMemo.contains(tableIR)

  override def toString: String = distinctMemo.toString
}
