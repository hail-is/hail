package org.broadinstitute.hail.driver

import org.apache.spark.{Accumulator, AccumulatorParam}
import org.apache.spark.AccumulatorParam._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Variant
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object Deduplicate extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "deduplicate"

  def description = "Remove duplicate variants from this dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  object DuplicateAccumulator extends AccumulatorParam[(Long, mutable.Set[Variant])] {
    def addInPlace(t1: (Long, mutable.Set[Variant]), t2: (Long, mutable.Set[Variant])): (Long, mutable.Set[Variant]) = {
      val (count1, set1) = t1
      val (count2, set2) = t2
      val set = if (set1.size >= 10)
        set1
      else if (set2.size >= 10)
        set2
      else {
        (set1 ++= set2).take(10)
      }
      (count1 + count2, set)
    }

    def zero(initialValue: (Long, mutable.Set[Variant])): (Long, mutable.Set[Variant]) = (0L, mutable.Set.empty[Variant])
  }


  object DeduplicateReport {

    var accumulator: Accumulator[(Long, mutable.Set[Variant])] = _

    def initialize() {
      accumulator = new Accumulator[(Long, mutable.Set[Variant])]((0L, mutable.Set.empty[Variant]), DuplicateAccumulator)
    }

    def report() {

      Option(accumulator).foreach { accumulator =>
        val (count, variants) = accumulator.value
        if (count > 0) {
          info(s"filtered $count duplicate variants")
          info(
            s"""Select duplicated variants:
                |  ${ variants.toArray.sorted.mkString(", ") }""".stripMargin)
        } else
          info("no duplicate variants found")
      }
    }
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    DeduplicateReport.initialize()

    val acc = DeduplicateReport.accumulator
    state.copy(vds = vds.copy(rdd = vds.rdd.mapPartitions({ it =>
      new SortedDistinctPairIterator(it, (v: Variant) => acc += (1L, mutable.Set(v)))
    }, preservesPartitioning = true).asOrderedRDD))
  }
}
