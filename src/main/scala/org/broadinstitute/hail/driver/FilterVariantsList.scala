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

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

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

    val in = vds.rdd
      .map { case (v, va, gs) => (v, (va, gs.toGenotypeStream(v, compress = false))) }

    state.copy(
      vds = vds.copy(
        rdd =
          if (keep)
            in
              .joinDistinct(variants)
              .map { case (v, ((va, gs), _)) => (v, va, gs) }
          else
            in
              .leftOuterJoinDistinct(variants)
              .flatMap {
                case (v, ((va, gs), Some(_))) =>
                  None
                case (v, ((va, gs), None)) =>
                  Some((v, va, gs))
              }
      ))
  }
}
