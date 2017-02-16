package is.hail.utils.richUtils

import breeze.linalg.{Vector => BVector}
import is.hail.utils._
import org.apache.spark.mllib.linalg.distributed.IndexedRow

import scala.language.{higherKinds, implicitConversions}

class RichIndexedRow(val r: IndexedRow) extends AnyVal {

  def -(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector - that)

  def +(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector + that)

  def :*(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector :* that)

  def :/(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector :/ that)
}
