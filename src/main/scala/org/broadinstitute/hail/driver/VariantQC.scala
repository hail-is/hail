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
  val header = "nCalled\t" +
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
    "MAF\t" +
    "nNonRef\t" +
    "rHeterozygosity\t" +
    "rHetHomVar\t" +
    "rExpectedHetFrequency\tpHWE\t"

  val signatures = Map("nCalled" -> new VCFSignature("Int", "toInt", ""),
    "nNotCalled" -> new VCFSignature("Int", "toInt", ""),
    "nHomRef" -> new VCFSignature("Int", "toInt", ""),
    "nHet" -> new VCFSignature("Int", "toInt", ""),
    "nHomVar" -> new VCFSignature("Int", "toInt", ""),
    "dpMean" -> new VCFSignature("Double", "toDouble", ""),
    "dpStDev" -> new VCFSignature("Double", "toDouble", ""),
    "dpMeanHomRef" -> new VCFSignature("Double", "toDouble", ""),
    "dpStDevHomRef" -> new VCFSignature("Double", "toDouble", ""),
    "dpMeanHet" -> new VCFSignature("Double", "toDouble", ""),
    "dpStDevHet" -> new VCFSignature("Double", "toDouble", ""),
    "dpMeanHomVar" -> new VCFSignature("Double", "toDouble", ""),
    "dpStDevHomVar" -> new VCFSignature("Double", "toDouble", ""),
    "gqMean" -> new VCFSignature("Double", "toDouble", ""),
    "gqStDev" -> new VCFSignature("Double", "toDouble", ""),
    "gqMeanHomRef" -> new VCFSignature("Double", "toDouble", ""),
    "gqStDevHomRef" -> new VCFSignature("Double", "toDouble", ""),
    "gqMeanHet" -> new VCFSignature("Double", "toDouble", ""),
    "gqStDevHet" -> new VCFSignature("Double", "toDouble", ""),
    "gqMeanHomVar" -> new VCFSignature("Double", "toDouble", ""),
    "gqStDevHomVar" -> new VCFSignature("Double", "toDouble", ""),
    "MAF" -> new VCFSignature("Double", "toDouble", ""),
    "nNonRef" -> new VCFSignature("Int", "toInt", ""),
    "rHeterozygosity" -> new VCFSignature("Double", "toDouble", ""),
    "rHetHomVar" -> new VCFSignature("Double", "toDouble", ""),
    "rExpectedHetFrequency" -> new VCFSignature("Double", "toDouble", ""),
    "pHWE" -> new VCFSignature("Double", "toDouble", ""))
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

    // MAF
    val refAlleles = nHomRef * 2 + nHet
    val altAlleles = nHomVar * 2 + nHet
    sb.tsvAppend(divOption(altAlleles, refAlleles + altAlleles))
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

    val hwe = HWEStats
    sb.tsvAppend(hwe._1)
    sb += '\t'
    sb ++= stringFormatDouble(hwe._2)
  }

  def asMap: Map[String, String] = {
    val maf = {
      val refAlleles = nHomRef * 2 + nHet
      val altAlleles = nHomVar * 2 + nHet
      divOption(altAlleles, refAlleles + altAlleles)}

    val hwe = HWEStats

    Map("nCalled" -> (nHomRef + nHet + nHomVar),
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
      "MAF" -> maf,
      "nNonRef" -> (nHet + nHomVar),
      "rHeterozygosity" -> divOption(nHet, nHomRef + nHet + nHomVar),
      "rHetHomVar" -> divOption(nHet, nHomVar),
      "rExpectedHetFrequency" -> hwe._1,
      "pHWE" -> hwe._2)
      .flatMap { case (k, v) => v match {
        case Some(value) => Some(k, value.toString)
        case None => None
        case _ => Some(k, v.toString)
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
        (ad: AnnotationData, comb: VariantQCCombiner) => ad.addMap("qc", comb.asMap))
        .addVariantSignatures(Map("qc" -> VariantQCCombiner.signatures)))
    else {
      writeTextFile(output + ".header", state.hadoopConf) { s =>
        s.write("Chrom\tPos\tRef\tAlt\t")
        s.write(VariantQCCombiner.header)
        s.write("\n")
      }

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
        }.saveAsTextFile(output)

      state
    }
  }
}
