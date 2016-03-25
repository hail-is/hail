package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable.ArrayBuffer

object FilterVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition: expression or .interval_list file")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "filtervariants"

  def description = "Filter variants in current dataset"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val vas = vds.vaSignature
    val cond = options.condition
    val keep = options.keep
    val p: (Variant, Annotation, Iterable[Genotype]) => Boolean = cond match {
      case f if f.endsWith(".interval_list") =>
        val ilist = IntervalList.read(options.condition, state.hadoopConf)
        val ilistBc = state.sc.broadcast(ilist)
        (v: Variant, va: Annotation, gs: Iterable[Genotype]) =>
          Filter.keepThis(ilistBc.value.contains(v.contig, v.start), keep)
      case c: String =>
        val symTab = Map(
          "v" ->(0, TVariant),
          "va" ->(1, vas),
          "gs" ->(2, TGenotypeStream))
        val symTab2 = Map(
          "v" ->(0, TVariant),
          "va" ->(1, vas),
          "s" ->(2, TSample),
          "sa" ->(3, vds.saSignature),
          "g" ->(4, TGenotype)
        )
        val a = new ArrayBuffer[Any]()
        val a2 = new ArrayBuffer[Any]()
        val a3 = new ArrayBuffer[Aggregator]()
        for (_ <- symTab)
          a += null
        for (_ <- symTab2)
          a2 += null
        val f: () => Any = Parser.parse[Any](symTab, symTab2, TBoolean, a, a2, a3, options.condition)
        for (_ <- a3)
          a2 += null
        val sampleInfoBc = vds.sparkContext.broadcast(
          vds.localSamples.map(vds.sampleAnnotations)
            .zip(vds.localSamples.map(vds.sampleIds).map(Sample)))
        (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
          a(0) = v
          a(1) = va
          a(2) = gs

          val computations = a3.toArray.map(_._1)
          gs.iterator
            .zip(sampleInfoBc.value.iterator)
            .foreach {
              case (g, (sa, s)) =>
                a2(0) = v
                a2(1) = va
                a2(2) = s
                a2(3) = sa
                a2(4) = g
                a3.iterator.zipWithIndex
                  .foreach {
                    case ((zv, so, co), i) =>
                      computations(i) = so(computations(i))
                  }
            }
          Filter.keepThisAny(f(), keep)
        }
    }

    state.copy(vds = vds.filterVariants(p))
  }
}
