package org.broadinstitute.hail.driver

import org.apache.commons.math3.distribution.BinomialDistribution
import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.stats.LeveneHaldane
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object VariantQCCombiner {
  val header = "callRate\tMAC\tMAF\tnCalled\t" +
    "nNotCalled\t" +
    "nHomRef\t" +
    "nHet\t" +
    "nHomVar\t" +
    "dpMean\tdpStDev\t" +
    "dpMeanHomRef\tdpStDevHomRef\t" +
    "dpMeanHet\tdpStDevHet\t" +
    "dpMeanHomVar\tdpStDevHomVar\t" +
    "gqMean\tgqStDev\t" +
    "gqMeanHomRef\tgqStDevHomRef\t" +
    "gqMeanHet\tgqStDevHet\t" +
    "gqMeanHomVar\tgqStDevHomVar\t" +
    "nNonRef\t" +
    "rHeterozygosity\t" +
    "rHetHomVar\t" +
    "rExpectedHetFrequency\tpHWE"

  val signatures = Map("callRate" -> new SimpleSignature("Double"),
    "MAC" -> new SimpleSignature("Int"),
    "MAF" -> new SimpleSignature("Double"),
    "nCalled" -> new SimpleSignature("Int"),
    "nNotCalled" -> new SimpleSignature("Int"),
    "nHomRef" -> new SimpleSignature("Int"),
    "nHet" -> new SimpleSignature("Int"),
    "nHomVar" -> new SimpleSignature("Int"),
    "dpMean" -> new SimpleSignature("Double"),
    "dpStDev" -> new SimpleSignature("Double"),
    "dpMeanHomRef" -> new SimpleSignature("Double"),
    "dpStDevHomRef" -> new SimpleSignature("Double"),
    "dpMeanHet" -> new SimpleSignature("Double"),
    "dpStDevHet" -> new SimpleSignature("Double"),
    "dpMeanHomVar" -> new SimpleSignature("Double"),
    "dpStDevHomVar" -> new SimpleSignature("Double"),
    "gqMean" -> new SimpleSignature("Double"),
    "gqStDev" -> new SimpleSignature("Double"),
    "gqMeanHomRef" -> new SimpleSignature("Double"),
    "gqStDevHomRef" -> new SimpleSignature("Double"),
    "gqMeanHet" -> new SimpleSignature("Double"),
    "gqStDevHet" -> new SimpleSignature("Double"),
    "gqMeanHomVar" -> new SimpleSignature("Double"),
    "gqStDevHomVar" -> new SimpleSignature("Double"),
    "nNonRef" -> new SimpleSignature("Int"),
    "rHeterozygosity" -> new SimpleSignature("Double"),
    "rHetHomVar" -> new SimpleSignature("Double"),
    "rExpectedHetFrequency" -> new SimpleSignature("Double"),
    "pHWE" -> new SimpleSignature("Double"))
}

class VariantQCCombiner extends Serializable {
  var nNotCalled: Int = 0
  var nHomRef: Int = 0
  var nHet: Int = 0
  var nHomVar: Int = 0
  var refDepth: Int = 0
  var altDepth: Int = 0

  val dpSC = new StatCounter()
  val dpHomRefSC = new StatCounter()
  val dpHetSC = new StatCounter()
  val dpHomVarSC = new StatCounter()

  val gqSC: StatCounter = new StatCounter()
  val gqHomRefSC: StatCounter = new StatCounter()
  val gqHetSC: StatCounter = new StatCounter()
  val gqHomVarSC: StatCounter = new StatCounter()

  // FIXME per-genotype

  def merge(g: Genotype): VariantQCCombiner = {
    (g.call.map(_.gt): @unchecked) match {
      case Some(0) =>
        nHomRef += 1
        dpSC.merge(g.dp)
        dpHomRefSC.merge(g.dp)
        gqSC.merge(g.gq)
        gqHomRefSC.merge(g.gq)
      case Some(1) =>
        nHet += 1
        refDepth += g.ad._1
        altDepth += g.ad._2
        dpSC.merge(g.dp)
        dpHetSC.merge(g.dp)
        gqSC.merge(g.gq)
        gqHetSC.merge(g.gq)
      case Some(2) =>
        nHomVar += 1
        dpSC.merge(g.dp)
        dpHomVarSC.merge(g.dp)
        gqSC.merge(g.gq)
        gqHomVarSC.merge(g.gq)
      case None =>
        nNotCalled += 1
    }

    this
  }

  def merge(that: VariantQCCombiner): VariantQCCombiner = {
    nNotCalled += that.nNotCalled
    nHomRef += that.nHomRef
    nHet += that.nHet
    nHomVar += that.nHomVar
    refDepth += that.refDepth
    altDepth += that.altDepth

    dpSC.merge(that.dpSC)
    dpHomRefSC.merge(that.dpHomRefSC)
    dpHetSC.merge(that.dpHetSC)
    dpHomVarSC.merge(that.dpHomVarSC)

    gqSC.merge(that.gqSC)
    gqHomRefSC.merge(that.gqHomRefSC)
    gqHetSC.merge(that.gqHetSC)
    gqHomVarSC.merge(that.gqHomVarSC)

    this
  }

  def emitSC(sb: mutable.StringBuilder, sc: StatCounter) {
    sb.tsvAppend(someIf(sc.count > 0, sc.mean))
    sb += '\t'
    sb.tsvAppend(someIf(sc.count > 0, sc.stdev))
  }

  def HWEStats: (Option[Double], Double) = {
    // rExpectedHetFrequency, pHWE
    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    (divOption(LH.getNumericalMean, n), LH.exactMidP(nAB))
  }

  def emit(sb: mutable.StringBuilder) {
    val nCalled = nHomRef + nHet + nHomVar

    val callRate = divOption(nCalled, nCalled + nNotCalled)
    val mac = nHet + 2 * nHomVar

    sb.tsvAppend(callRate)
    sb += '\t'
    sb.append(mac)
    sb += '\t'
    // MAF
    val refAlleles = nHomRef * 2 + nHet
    val altAlleles = nHomVar * 2 + nHet
    sb.tsvAppend(divOption(altAlleles, refAlleles + altAlleles))
    sb += '\t'
    sb.append(nCalled)
    sb += '\t'
    sb.append(nNotCalled)
    sb += '\t'
    sb.append(nHomRef)
    sb += '\t'
    sb.append(nHet)
    sb += '\t'
    sb.append(nHomVar)
    sb += '\t'

    emitSC(sb, dpSC)
    sb += '\t'
    emitSC(sb, dpHomRefSC)
    sb += '\t'
    emitSC(sb, dpHetSC)
    sb += '\t'
    emitSC(sb, dpHomVarSC)
    sb += '\t'

    emitSC(sb, gqSC)
    sb += '\t'
    emitSC(sb, gqHomRefSC)
    sb += '\t'
    emitSC(sb, gqHetSC)
    sb += '\t'
    emitSC(sb, gqHomVarSC)
    sb += '\t'

    // nNonRef
    sb.append(nHet + nHomVar)
    sb += '\t'

    // rHeterozygosity
    sb.tsvAppend(divOption(nHet, nCalled))
    sb += '\t'

    // rHetHomVar
    sb.tsvAppend(divOption(nHet, nHomVar))
    sb += '\t'

    // Hardy-Weinberg statistics
    val hwe = HWEStats
    sb.tsvAppend(hwe._1)
    sb += '\t'
    sb.tsvAppend(hwe._2)
  }

  def asMap: Map[String, Any] = {
    val maf = {
      val refAlleles = nHomRef * 2 + nHet
      val altAlleles = nHomVar * 2 + nHet
      divOption(altAlleles, refAlleles + altAlleles)}

    val nCalled = nHomRef + nHet + nHomVar
    val hwe = HWEStats
    val callrate = divOption(nCalled, nCalled + nNotCalled)
    val mac = nHet + 2 * nHomVar

    Map[String, Any]("callRate" -> divOption(nCalled, nCalled + nNotCalled),
      "MAC" -> mac,
      "MAF" -> maf,
      "nCalled" -> nCalled,
      "nNotCalled" -> nNotCalled,
      "nHomRef" -> nHomRef,
      "nHet" -> nHet,
      "nHomVar" -> nHomVar,
      "dpMean" -> someIf(dpSC.count > 0, dpSC.mean),
      "dpStDev" -> someIf(dpSC.count > 0, dpSC.stdev),
      "dpMeanHomRef" -> someIf(dpHomRefSC.count > 0, dpHomRefSC.mean),
      "dpStDevHomRef" -> someIf(dpHomRefSC.count > 0, dpHomRefSC.stdev),
      "dpMeanHet" -> someIf(dpHetSC.count > 0, dpHetSC.mean),
      "dpStDevHet" -> someIf(dpHetSC.count > 0, dpHetSC.stdev),
      "dpMeanHomVar" -> someIf(dpHomVarSC.count > 0, dpHomVarSC.mean),
      "dpStDevHomVar" -> someIf(dpHomVarSC.count > 0, dpHomVarSC.stdev),
      "gqMean" -> someIf(gqSC.count > 0, gqSC.mean),
      "gqStDev" -> someIf(gqSC.count > 0, gqSC.stdev),
      "gqMeanHomRef" -> someIf(gqHomRefSC.count > 0, gqHomRefSC.mean),
      "gqStDevHomRef" -> someIf(gqHomRefSC.count > 0, gqHomRefSC.stdev),
      "gqMeanHet" -> someIf(gqHetSC.count > 0, gqHetSC.mean),
      "gqStDevHet" -> someIf(gqHetSC.count > 0, gqHetSC.stdev),
      "gqMeanHomVar" -> someIf(gqHomVarSC.count > 0, gqHomVarSC.mean),
      "gqStDevHomVar" -> someIf(gqHomVarSC.count > 0, gqHomVarSC.stdev),
      "nNonRef" -> (nHet + nHomVar),
      "rHeterozygosity" -> divOption(nHet, nHomRef + nHet + nHomVar),
      "rHetHomVar" -> divOption(nHet, nHomVar),
      "rExpectedHetFrequency" -> hwe._1,
      "pHWE" -> hwe._2)
      .flatMap { case (k, v) => v match {
        case Some(value) => Some(k, value)
        case None => None
        case _ => Some(k, v)
      }}
  }
}

object VariantQC extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-o", aliases = Array("--output"),
      usage = "Output file", forbids = Array("store"))
    var output: String = ""

    @Args4jOption(required = false, name = "-s", aliases = Array("--store"),
      usage = "Store qc output in vds annotations", forbids = Array("output"))
    var store: Boolean = false
  }

  def newOptions = new Options

  def name = "variantqc"

  def description = "Compute per-variant QC metrics"

  def results(vds: VariantDataset): RDD[(Variant, VariantQCCombiner)] =
    vds
      .aggregateByVariant(new VariantQCCombiner)((comb, g) => comb.merge(g),
        (comb1, comb2) => comb1.merge(comb2))


  def run(state: State, options: Options): State = {
    val vds = state.vds

    val output = options.output

    if (options.store)
      state.copy(vds = vds.mapAnnotationsWithAggregate(new VariantQCCombiner)((comb, v, s, g) => comb.merge(g),
        (comb1, comb2) => comb1.merge(comb2),
        (va: Annotations, comb: VariantQCCombiner) => va + ("qc", Annotations(comb.asMap)))
        .addVariantAnnotationSignatures("qc", Annotations(VariantQCCombiner.signatures)))
    else {
      val qcResults = results(vds)

      hadoopDelete(output, state.hadoopConf, recursive = true)
      val r = results(vds)
        .map { case (v, comb) =>
          val sb = new StringBuilder()
          sb.append(v.contig)
          sb += '\t'
          sb.append(v.start)
          sb += '\t'
          sb.append(v.ref)
          sb += '\t'
          sb.append(v.alt)
          sb += '\t'
          comb.emit(sb)
          sb.result()
        }.writeTable(output, Some("Chrom\tPos\tRef\tAlt\t" + VariantQCCombiner.header))

      state
    }
  }
}
