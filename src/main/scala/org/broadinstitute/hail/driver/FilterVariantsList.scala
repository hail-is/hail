package org.broadinstitute.hail.driver

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariantsList extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-i", aliases = Array("--input"),
      usage = "Path to variant list file")
    var input: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false
  }

  def newOptions = new Options

  def name = "filtervariants list"

  def description = "Filter variants in current dataset with a variant list"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val keep = options.keep

    val variants: RDD[(Variant, Unit)] =
      vds.sparkContext.textFileLines(options.input)
        .map {
          _.map { line =>
            val fields = line.split(":")
            if (fields.length != 4)
              fatal("invalid variant: expect `CHR:POS:REF:ALT1,ALT2,...,ALTN'")
            val ref = fields(2)
            (Variant(fields(0),
              fields(1).toInt,
              ref,
              fields(3).split(",").map(alt => AltAllele(ref, alt))), ())
          }.value
        }

    state.copy(
      vds = vds.copy(
        rdd = vds.rdd
          .orderedLeftJoinDistinct(variants.toOrderedRDD(_.locus))
          .mapPartitions({ it =>
            it.flatMap { case (v, ((va, gs), o)) =>
              o match {
                case Some(_) =>
                  if (keep) Some((v, (va, gs))) else None
                case None =>
                  if (keep) None else Some((v, (va, gs)))
              }
            }
          }, preservesPartitioning = true)
          .toOrderedRDD(_.locus)
      ))
  }
}
