package is.hail.expr.ir

import is.hail.utils.TreeTraversal

object DistinctlyKeyed {

  def apply(node: BaseIR): DistinctKeyedAnalysis =
    DistinctKeyedAnalysis(flattenIR(node).foldLeft(Memo.empty[Unit])(analyze))

  /**
   * Update the memo if the node is distinctly-keyed
   */
  private def analyze(memo: Memo[Unit], node: BaseIR): Memo[Unit] = node match {
    case t: TableRead =>
      memo.bindIf(t.isDistinctlyKeyed, t, ())

    case t@TableKeyBy(child, keys, _) =>
      memo.bindIf(child.typ.key.forall(keys.contains) && memo.contains(child), t, ())

    case t@(_: TableFilter | _: TableMapRows | _: TableMapGlobals) =>
      memo.bindIf(memo.contains(t.children.head), t, ())

    case t@RelationalLetTable(_, _, body) =>
      memo.bindIf(memo.contains(body), t, ())

    case t@(_: TableHead
            | _: TableTail
            | _: TableRepartition
            | _: TableJoin
            | _: TableIntervalJoin
            | _: TableMultiWayZipJoin
            | _: TableLeftJoinRightDistinct
            | _: TableRename
            | _: TableFilterIntervals) =>
      memo.bindIf(t.children.forall(memo.contains), t, ())

    case t@(_: TableRange
            | _: TableDistinct
            | _: TableKeyByAndAggregate
            | _: TableAggregateByKey) =>
      memo.bind(t, ())

    case _: MatrixIR =>
      throw new IllegalArgumentException("MatrixIR should be lowered when it reaches distinct analysis")

    case _ =>
      memo
  }

  /**
   * Iterate over the "interesting" nodes in the IR in post-order (ie. only
   * those nodes that contribute to the analysis of distinctly-keyed nodes).
   */
  private def flattenIR: BaseIR => Iterator[BaseIR] =
    TreeTraversal.postOrder {
      case t@(_: TableAggregateByKey
              | _: TableFilter
              | _: TableKeyByAndAggregate
              | _: TableMapGlobals
              | _: TableMapPartitions
              | _: TableMapRows) =>
        t.children.take(1).iterator

      case RelationalLetTable(_, _, body) =>
        Iterator.single(body)

      case t@(_: IR
              | _: TableDistinct
              | _: TableExplode
              | _: TableFilterIntervals
              | _: TableHead
              | _: TableIntervalJoin
              | _: TableJoin
              | _: TableKeyBy
              | _: TableLeftJoinRightDistinct
              | _: TableMultiWayZipJoin
              | _: TableOrderBy
              | _: TableRepartition
              | _: TableRename
              | _: TableToTableApply
              | _: TableTail
              | _: TableUnion) =>
        t.children.iterator

      case _ =>
        Iterator.empty
    }
}

case class DistinctKeyedAnalysis(distinctMemo: Memo[Unit]) {
  def contains(tableIR: BaseIR): Boolean = {
    distinctMemo.contains(tableIR)
  }

  override def toString: String = distinctMemo.toString
}