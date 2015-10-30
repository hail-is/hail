package org.broadinstitute.hail.driver

import org.apache.commons.math3.distribution.BinomialDistribution
import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object VariantQCCombiner {
  val header = "nCalled" + "\t" +
    "nNotCalled" + "\t" +
    "nHomRef" + "\t" +
    "nHet" + "\t" +
    "nHomVar" + "\t" +
    "alleleBalance" + "\t" +
    "dpMean" + "\t" + "dpStDev" + "\t" +
    "dpHomRefMean" + "\t" + "dpHomRefStDev" + "\t" +
    "dpHetMean" + "\t" + "dpHetStDev" + "\t" +
    "dpHomVarMean" + "\t" + "dpHomVarStDev" + "\t" +
    "gqMean" + "\t" + "gqStDev" + "\t" +
    "gqHomRefMean" + "\t" + "gqHomRefStDev" + "\t" +
    "gqHetMean" + "\t" + "gqHetStDev" + "\t" +
    "gqHomVarMean" + "\t" + "gqHomVarStDev" + "\t" +
    "MAF" + "\t" +
    "nNonRef" + "\t" +
    "rHeterozygosity" + "\t" +
    "rHetHomVar" + "\t" +
    "pHWE"
}

class VariantQCCombiner {
  private var nCalled: Int = 0
  private var nNotCalled: Int = 0
  private var nHomRef: Int = 0
  private var nHet: Int = 0
  private var nHomVar: Int = 0
  private var refDepth: Int = 0
  private var altDepth: Int = 0

  private val dpSC = new StatCounter()
  private val dpHomRefSC = new StatCounter()
  private val dpHetSC = new StatCounter()
  private val dpHomVarSC = new StatCounter()

  private val gqSC: StatCounter = new StatCounter()
  private val gqHomRefSC: StatCounter = new StatCounter()
  private val gqHetSC: StatCounter = new StatCounter()
  private val gqHomVarSC: StatCounter = new StatCounter()

  // FIXME per-genotype

  def merge(g: Genotype): VariantQCCombiner = {
    g.call.map(_.gt) match {
      case Some(0) =>
        nHomRef += 1
        nCalled += 1
        dpSC.merge(g.dp)
        dpHomRefSC.merge(g.dp)
        gqSC.merge(g.gq)
        gqHomRefSC.merge(g.gq)
      case Some(1) =>
        nHet += 1
        nCalled += 1
        refDepth += g.ad._1
        altDepth += g.ad._2
        dpSC.merge(g.dp)
        dpHetSC.merge(g.dp)
        gqSC.merge(g.gq)
        gqHetSC.merge(g.gq)
      case Some(2) =>
        nHomVar += 1
        nCalled += 1
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
    nCalled += that.nCalled
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

  def pAB: Double = {
    val d = new BinomialDistribution(refDepth + altDepth, 0.5)
    val minDepth = refDepth.min(altDepth)
    val minp = d.probability(minDepth)
    val mincp = d.cumulativeProbability(minDepth)
    (2 * mincp - minp).min(1.0).max(0.0)
  }

  def pHWE: Double = {
    val total = nHomRef + nHet + nHomVar
    val p = (nHet.toDouble + 2 * nHomVar) / total
    val q = 1 - p

    val observed = Array[Double](nHomRef, nHet, nHomVar)
    val expected = Array[Double](q * q * total, 2 * p * q * total, p * p * total)

    // FIXME handle div by 0
    observed.zipWith[Double, Double](expected, (o, e) => (o - e) * (o - e) / e).sum
  }

  def emitSC(sb: mutable.StringBuilder, sc: StatCounter) {
    sb.append(toTSVString(someIf(sc.count > 0, sc.mean)))
    sb += '\t'
    sb.append(toTSVString(someIf(sc.count > 0, sc.stdev)))
  }

  def emit(sb: mutable.StringBuilder) {
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

    sb.append(pAB)
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
    sb.append(toTSVString(divOption(altAlleles, refAlleles + altAlleles)))
    sb += '\t'

    // nNonRef
    sb.append(nHet + nHomVar)
    sb += '\t'

    // rHeterozygosity
    sb.append(toTSVString(divOption(nHet, nCalled)))
    sb += '\t'

    // rHetHomVar
    sb.append(toTSVString(divOption(nHet, nHomVar)))
    sb += '\t'

    // pHWE
    sb.append(pHWE)
  }
}

object VariantQC2 extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "variantqc2"

  def description = "Compute per-variant QC metrics"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val output = options.output

    writeTextFile(output + ".header", state.hadoopConf) { s =>
      s.write("Chrom\tPos\tRef\tAlt\t")
      s.write(VariantQCCombiner.header)
      s.write("\n")
    }

    val r = vds
      .aggregateByVariant(new VariantQCCombiner)((comb, g) => comb.merge(g),
        (comb1, comb2) => comb1.merge(comb2))
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

object VariantQC extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "variantqc"

  def description = "Compute per-variant QC metrics"

  def results(vds: VariantDataset,
    methods: Array[AggregateMethod],
    derivedMethods: Array[DerivedMethod] = Array()): RDD[(Variant, Array[Any])] = {

    val methodIndex = methods.zipWithIndex.toMap

    val methodsBc = vds.sparkContext.broadcast(methods)

    vds
      .aggregateByVariantWithKeys(methods.map(_.aggZeroValue: Any))(
        (acc, v, s, g) => methodsBc.value.zipWith[Any, Any](acc, (m, acci) =>
            m.seqOpWithKeys(v, s, g, acci.asInstanceOf[m.T])),
        (x, y) => methodsBc.value.zipWith[Any, Any, Any](x, y, (m, xi, yi) =>
            m.combOp(xi.asInstanceOf[m.T], yi.asInstanceOf[m.T])))
      .mapValues(values => {
        val b = mutable.ArrayBuilder.make[Any]()
        values.foreach2[AggregateMethod](methodsBc.value, (v, m) => m.emit(v.asInstanceOf[m.T], b))
        val methodValues = MethodValues(methodIndex, values)
        derivedMethods.foreach(_.emit(methodValues, b))
        b.result()
      })
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val methods: Array[AggregateMethod] = Array(
      nCalledPer, nNotCalledPer,
      nHomRefPer, nHetPer, nHomVarPer,
      AlleleBalancePer, dpStatCounterPer, dpStatCounterPerGenotype, gqStatCounterPer, gqStatCounterPerGenotype
    )

    val derivedMethods: Array[DerivedMethod] = Array(
      minorAlleleFrequencyPer, nNonRefPer, rHeterozygosityPer, rHetHomVarPer, pHwePerVariant
    )

    val r = results(vds, methods, derivedMethods)

    val allMethods = methods ++ derivedMethods

    writeTextFile(options.output + ".header", state.hadoopConf) { s =>
      val header = "Chrom" + "\t" + "Pos" + "\t" + "Ref" + "\t" + "Alt" + "\t" +
        allMethods.map(_.name).filter(_ != null).mkString("\t") + "\n"
      s.write(header)
    }

    val output = options.output

    hadoopDelete(output, vds.sparkContext.hadoopConfiguration, true)
    r.map { case (v, a) =>
      (Array[Any](v.contig, v.start, v.ref, v.alt) ++ a).map(toTSVString).mkString("\t")
    }.saveAsTextFile(output)

    state
  }
}
