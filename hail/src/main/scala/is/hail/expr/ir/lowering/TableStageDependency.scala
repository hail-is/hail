package is.hail.expr.ir.lowering

import is.hail.rvd.RVD
import is.hail.utils.FastIndexedSeq

trait DependencySource

case class RVDDependency(rvd: RVD) extends DependencySource

object TableStageDependency {
  val none: TableStageDependency = TableStageDependency(FastIndexedSeq())

  def union(others: IndexedSeq[TableStageDependency]): TableStageDependency = {
    assert(others.nonEmpty)

    TableStageDependency(others.flatMap(_.deps))
  }

  def fromRVD(rvd: RVD): TableStageDependency =
    TableStageDependency(FastIndexedSeq(RVDDependency(rvd)))
}

case class TableStageDependency(deps: IndexedSeq[DependencySource]) {
  def union(other: TableStageDependency): TableStageDependency = TableStageDependency.union(FastIndexedSeq(this, other))
}
