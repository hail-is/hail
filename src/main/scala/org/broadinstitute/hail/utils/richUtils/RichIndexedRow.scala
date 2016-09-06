package org.broadinstitute.hail.utils.richUtils

import breeze.linalg.{Vector => BVector}
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.broadinstitute.hail.utils._

import scala.language.{higherKinds, implicitConversions}

class RichIndexedRow(val r: IndexedRow) extends AnyVal {

  def -(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector - that)

  def +(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector + that)

  def :*(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector :* that)

  def :/(that: BVector[Double]): IndexedRow = IndexedRow(r.index, r.vector :/ that)
}
