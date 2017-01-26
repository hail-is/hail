package is.hail.driver

import is.hail.utils._
import is.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object Join extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-r", aliases = Array("--right"), usage = "Name of dataset in environment to join on the right")
    var rightName: String = _
  }

  def newOptions = new Options

  def name = "join"

  def description = "Join datasets, inner join on variants, concatenate samples, variant and global annotations from left"

  def supportsMultiallelic = true

  def requiresVDS = true

  def join(left: VariantDataset, right: VariantDataset): VariantDataset = {
    if (left.wasSplit != right.wasSplit) {
      warn(
        s"""joining split and unsplit datasets may result in unexpected behavior
            |  left was split: ${ left.wasSplit }
            |  light was split: ${ right.wasSplit }""".stripMargin)
    }

    if (left.saSignature != right.saSignature) {
      fatal(
        s"""cannot join datasets with different sample schemata
            |  left sample schema: @1
            |  right sample schema: @2""".stripMargin,
        left.saSignature.toPrettyString(compact = true, printAttrs = true),
        right.saSignature.toPrettyString(compact = true, printAttrs = true))
    }

    val newSampleIds = left.sampleIds ++ right.sampleIds
    val duplicates = newSampleIds.duplicates()
    if (duplicates.nonEmpty)
      fatal("duplicate sample IDs: @1", duplicates)

    val joined = left.rdd.orderedInnerJoinDistinct(right.rdd)
      .mapValues { case ((lva, lgs), (rva, rgs)) =>
        (lva, lgs ++ rgs)
      }.asOrderedRDD

    left.copy(
      sampleIds = newSampleIds,
      sampleAnnotations = left.sampleAnnotations ++ right.sampleAnnotations,
      rdd = joined)
  }

  def run(state: State, options: Options): State = {
    val left = state.vds
    val rightName = options.rightName

    val right = state.env.get(rightName) match {
      case Some(r) => r
      case None =>
        fatal(s"no such dataset $name in environment")
    }

    state.copy(
      vds = join(left, right))
  }
}
