package org.broadinstitute.hail.driver

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.expr._
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

  val signature = TStruct(
    "callRate" -> TDouble,
    "MAC" -> TInt,
    "MAF" -> TDouble,
    "nCalled" -> TInt,
    "nNotCalled" -> TInt,
    "nHomRef" -> TInt,
    "nHet" -> TInt,
    "nHomVar" -> TInt,
    "dpMean" -> TDouble,
    "dpStDev" -> TDouble,
    "dpMeanHomRef" -> TDouble,
    "dpStDevHomRef" -> TDouble,
    "dpMeanHet" -> TDouble,
    "dpStDevHet" -> TDouble,
    "dpMeanHomVar" -> TDouble,
    "dpStDevHomVar" -> TDouble,
    "gqMean" -> TDouble,
    "gqStDev" -> TDouble,
    "gqMeanHomRef" -> TDouble,
    "gqStDevHomRef" -> TDouble,
    "gqMeanHet" -> TDouble,
    "gqStDevHet" -> TDouble,
    "gqMeanHomVar" -> TDouble,
    "gqStDevHomVar" -> TDouble,
    "nNonRef" -> TInt,
    "rHeterozygosity" -> TDouble,
    "rHetHomVar" -> TDouble,
    "rExpectedHetFrequency" -> TDouble,
    "pHWE" -> TDouble)
}

class VariantQCCombiner extends Serializable {
  var nNotCalled: Int = 0
  var nHomRef: Int = 0
  var nHet: Int = 0
  var nHomVar: Int = 0

  val dpHomRefSC = new StatCounter()
  val dpHetSC = new StatCounter()
  val dpHomVarSC = new StatCounter()

  val gqHomRefSC: StatCounter = new StatCounter()
  val gqHetSC: StatCounter = new StatCounter()
  val gqHomVarSC: StatCounter = new StatCounter()

  def dpSC: StatCounter = {
    val r = dpHomRefSC.copy()
    r.merge(dpHetSC)
    r.merge(dpHomVarSC)
    r
  }

  def gqSC: StatCounter = {
    val r = gqHomRefSC.copy()
    r.merge(gqHetSC)
    r.merge(gqHomVarSC)
    r
  }

  // FIXME per-genotype

  def merge(g: Genotype): VariantQCCombiner = {
    (g.gt: @unchecked) match {
      case Some(0) =>
        nHomRef += 1
        g.dp.foreach { v =>
          dpHomRefSC.merge(v)
        }
        g.gq.foreach { v =>
          gqHomRefSC.merge(v)
        }
      case Some(1) =>
        nHet += 1
        g.dp.foreach { v =>
          dpHetSC.merge(v)
        }
        g.gq.foreach { v =>
          gqHetSC.merge(v)
        }
      case Some(2) =>
        nHomVar += 1
        g.dp.foreach { v =>
          dpHomVarSC.merge(v)
        }
        g.gq.foreach { v =>
          gqHomVarSC.merge(v)
        }
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

    dpHomRefSC.merge(that.dpHomRefSC)
    dpHetSC.merge(that.dpHetSC)
    dpHomVarSC.merge(that.dpHomVarSC)

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

  def asAnnotation: Annotation = {
    val maf = {
      val refAlleles = nHomRef * 2 + nHet
      val altAlleles = nHomVar * 2 + nHet
      divOption(altAlleles, refAlleles + altAlleles)
    }

    val nCalled = nHomRef + nHet + nHomVar
    val hwe = HWEStats
    val callrate = divOption(nCalled, nCalled + nNotCalled)
    val mac = nHet + 2 * nHomVar

    Annotation(
      divNull(nCalled, nCalled + nNotCalled),
      mac,
      maf.getOrElse(null),
      nCalled,
      nNotCalled,
      nHomRef,
      nHet,
      nHomVar,
      nullIfNot(dpSC.count > 0, dpSC.mean),
      nullIfNot(dpSC.count > 0, dpSC.stdev),
      nullIfNot(dpHomRefSC.count > 0, dpHomRefSC.mean),
      nullIfNot(dpHomRefSC.count > 0, dpHomRefSC.stdev),
      nullIfNot(dpHetSC.count > 0, dpHetSC.mean),
      nullIfNot(dpHetSC.count > 0, dpHetSC.stdev),
      nullIfNot(dpHomVarSC.count > 0, dpHomVarSC.mean),
      nullIfNot(dpHomVarSC.count > 0, dpHomVarSC.stdev),
      nullIfNot(gqSC.count > 0, gqSC.mean),
      nullIfNot(gqSC.count > 0, gqSC.stdev),
      nullIfNot(gqHomRefSC.count > 0, gqHomRefSC.mean),
      nullIfNot(gqHomRefSC.count > 0, gqHomRefSC.stdev),
      nullIfNot(gqHetSC.count > 0, gqHetSC.mean),
      nullIfNot(gqHetSC.count > 0, gqHetSC.stdev),
      nullIfNot(gqHomVarSC.count > 0, gqHomVarSC.mean),
      nullIfNot(gqHomVarSC.count > 0, gqHomVarSC.stdev),
      nHet + nHomVar,
      divNull(nHet, nHomRef + nHet + nHomVar),
      divNull(nHet, nHomVar),
      hwe._1.getOrElse(null),
      hwe._2)
  }
}

object VariantQC extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-o", aliases = Array("--output"),
      usage = "Output file")
    var output: String = _
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

    val r = results(vds).persist(StorageLevel.MEMORY_AND_DISK)


    if (output != null) {
      hadoopDelete(output, state.hadoopConf, recursive = true)
      r.map { case (v, comb) =>
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
    }

    val (newVAS, insertQC) = vds.vaSignature.insert(VariantQCCombiner.signature, "qc")
    state.copy(
      vds = vds.copy(
        rdd = vds.rdd.zipPartitions(r) { case (it, jt) =>
          it.zip(jt).map { case ((v, va, gs), (v2, comb)) =>
            assert(v == v2)
            (v, insertQC(va, Some(comb.asAnnotation)), gs)
          }
        },
        vaSignature = newVAS)
    )
  }
}
