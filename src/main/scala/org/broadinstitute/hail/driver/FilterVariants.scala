package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--all", usage = "Filter all variants")
    var all: Boolean = false

    @Args4jOption(required = false, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition: expression, .interval_list or .variant_list file")
    var condition: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

  }

  def newOptions = new Options

  def name = "filtervariants"

  def description = "Filter variants in current dataset"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

    if ((options.all && options.condition != null)
      || (!options.all && options.condition == null))
      fatal("one `--all' or `-c' required, but not both")

    if (options.all) {
      if (options.keep)
        return state
      else
        return state.copy(
          vds = state.vds.copy(rdd = state.sc.emptyRDD))
    }

    val vas = vds.vaSignature
    val cond = options.condition
    val keep = options.keep
    val p: (Variant, Annotation, Iterable[Genotype]) => Boolean = cond match {
      case f if f.endsWith(".interval_list") =>
        val ilist = IntervalList.read(options.condition, state.hadoopConf)
        val ilistBc = state.sc.broadcast(ilist)
        (v: Variant, _: Annotation, _: Iterable[Genotype]) =>
          Filter.keepThis(ilistBc.value.contains(v.contig, v.start), keep)

      case f if f.endsWith(".variant_list") =>
        val variants = readLines(f, state.hadoopConf)(_.map(_.transform { line =>
          val fields = line.value.split(":")
          if (fields.length != 4)
            fatal("invalid variant")
          val ref = fields(2)
          Variant(fields(0),
            fields(1).toInt,
            ref,
            fields(3).split(",").map(alt => AltAllele(ref, alt)))
        }).toSet)

        val variantsBc = state.sc.broadcast(variants)

        (v: Variant, _: Annotation, _: Iterable[Genotype]) => Filter.keepThis(variantsBc.value.contains(v), keep)

      case c: String =>
        val aggregationEC = EvalContext(Map(
          "v" ->(0, TVariant),
          "va" ->(1, vds.vaSignature),
          "s" ->(2, TSample),
          "sa" ->(3, vds.saSignature),
          "g" ->(4, TGenotype)
        ))
        val symTab = Map(
          "v" ->(0, TVariant),
          "va" ->(1, vds.vaSignature),
          "gs" ->(-1, TAggregable(aggregationEC)))

        val ec = EvalContext(symTab)
        val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

        val aggregatorOption = Aggregators.buildVariantaggregations(vds, aggregationEC)

        (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
          ec.setContext(v, va)

          aggregatorOption.foreach(f => f(v, va, gs))

          Filter.keepThis(f(), keep)
        }
    }

    state.copy(vds = vds.filterVariants(p))
  }
}
