package is.hail.expr.ir.lowering

import is.hail.rvd.RVD
import is.hail.utils.FastSeq

trait DependencySource

case class RVDDependency(rvd: RVD) extends DependencySource

object TableStageDependency {
  val none: TableStageDependency = TableStageDependency(FastSeq())

  def union(others: IndexedSeq[TableStageDependency]): TableStageDependency = {
    assert(others.nonEmpty)

    TableStageDependency(others.flatMap(_.deps))
  }

  def fromRVD(rvd: RVD): TableStageDependency =
    TableStageDependency(FastSeq(RVDDependency(rvd)))
}

case class TableStageDependency(deps: IndexedSeq[DependencySource]) {
  def union(other: TableStageDependency): TableStageDependency = TableStageDependency.union(FastSeq(this, other))
}
