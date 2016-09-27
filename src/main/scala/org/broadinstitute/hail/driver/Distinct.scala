package org.broadinstitute.hail.driver

import org.apache.spark.Accumulator
import org.apache.spark.AccumulatorParam._
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Distinct extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "deduplicate"

  def description = "Remove duplicate variants from this dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  object DistinctReport {

    var accumulator: Accumulator[Long] = _

    def initialize() {
      accumulator = new Accumulator[Long](0, LongAccumulatorParam)
    }

    def report() {

      Option(accumulator).foreach { accumulator =>
        if (accumulator.value > 0)
          info(s"filtered ${ accumulator.value } duplicate variants")
        else
          info("no duplicate variants found")
      }
    }
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    DistinctReport.initialize()

    val acc = DistinctReport.accumulator
    state.copy(vds = vds.copy(rdd = vds.rdd.mapPartitions({ it =>
      new SortedDistinctPairIterator(it, acc += 1L)
    }, preservesPartitioning = true).asOrderedRDD))
  }
}
