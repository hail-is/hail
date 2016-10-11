package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object Join extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-r", aliases = Array("--right"), usage = ".vds file to join on the right")
    var right: String = _
  }

  def newOptions = new Options

  def name = "join"

  def description = "Join datasets, inner join on variants, concatenate samples, variant and global annotations from left"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val left = state.vds
    val right = VariantSampleMatrix.read(state.sqlContext, options.right)

    if (left.wasSplit != right.wasSplit) {
      fatal(
        s"""cannot join split and unsplit datasets
            |  left was split: ${ left.wasSplit }
            |  light was split: ${ right.wasSplit }""".stripMargin)
    } else if (left.vaSignature != right.vaSignature) {
      fatal(
        s"""cannot join datasets with different variant schemata
            |  left variant schema: @1
            |  right variant schema: @2""".
          stripMargin,
        left.vaSignature.toPrettyString(compact = true, printAttrs = true),
        right.vaSignature.toPrettyString(compact = true, printAttrs = true))
    }

    val joined = left.rdd.orderedLeftJoinDistinct(right.rdd)
      .flatMapValues { case ((lva, lgs), None) =>
        None
      case ((lva, lgs), Some((rva, rgs))) =>
        Some((lva, lgs ++ rgs))
      }
      .toOrderedRDD

    var sampleSet = left.sampleIds.toSet
    assert(sampleSet.size == left.nSamples)

    val newRightSampleIds = right.sampleIds.map { s =>
      var s2 = s
      var changed = false
      while (sampleSet.contains(s2)) {
        s2 = s2 + "d"
        changed = true
      }
      if (changed)
        info(s"renamed right sampleId $s to $s2")
      sampleSet += s2
      s2
    }

    state.copy(
      vds = left.copy(
        sampleIds = left.sampleIds ++ newRightSampleIds,
        sampleAnnotations = left.sampleAnnotations ++ right.sampleAnnotations,
        rdd = joined))
  }
}
