package org.broadinstitute.hail.driver

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object SampleQCCombiner {
  val header = "callRate\t" +
    "nCalled\t" +
    "nNotCalled\t" +
    "nHomRef\t" +
    "nHet\t" +
    "nHomVar\t" +
    "nSNP\t" +
    "nInsertion\t" +
    "nDeletion\t" +
    "nSingleton\t" +
    "nTransition\t" +
    "nTransversion\t" +
    "dpMean\tdpStDev\t" +
    "dpMeanHomRef\tdpStDevHomRef\t" +
    "dpMeanHet\tdpStDevHet\t" +
    "dpMeanHomVar\tdpStDevHomVar\t" +
    "gqMean\tgqStDev\t" +
    "gqMeanHomRef\tgqStDevHomRef\t" +
    "gqMeanHet\tgqStDevHet\t" +
    "gqMeanHomVar\tgqStDevHomVar\t" +
    "nNonRef\t" +
    "rTiTv\t" +
    "rHetHomVar\t" +
    "rDeletionInsertion"

  val signatures: StructSignature = StructSignature(Map(
    "callRate" ->(0, new SimpleSignature(expr.TDouble)),
    "nCalled" ->(1, new SimpleSignature(expr.TInt)),
    "nNotCalled" ->(2, new SimpleSignature(expr.TInt)),
    "nHomRef" ->(3, new SimpleSignature(expr.TInt)),
    "nHet" ->(4, new SimpleSignature(expr.TInt)),
    "nHomVar" ->(5, new SimpleSignature(expr.TInt)),
    "nSNP" ->(6, new SimpleSignature(expr.TInt)),
    "nInsertion" ->(7, new SimpleSignature(expr.TInt)),
    "nDeletion" ->(8, new SimpleSignature(expr.TInt)),
    "nSingleton" ->(9, new SimpleSignature(expr.TInt)),
    "nTransition" ->(10, new SimpleSignature(expr.TInt)),
    "nTransversion" ->(11, new SimpleSignature(expr.TInt)),
    "dpMean" ->(12, new SimpleSignature(expr.TDouble)),
    "dpStDev" ->(13, new SimpleSignature(expr.TDouble)),
    "dpMeanHomRef" ->(14, new SimpleSignature(expr.TDouble)),
    "dpStDevHomRef" ->(15, new SimpleSignature(expr.TDouble)),
    "dpMeanHet" ->(16, new SimpleSignature(expr.TDouble)),
    "dpStDevHet" ->(17, new SimpleSignature(expr.TDouble)),
    "dpMeanHomVar" ->(18, new SimpleSignature(expr.TDouble)),
    "dpStDevHomVar" ->(19, new SimpleSignature(expr.TDouble)),
    "gqMean" ->(20, new SimpleSignature(expr.TDouble)),
    "gqStDev" ->(21, new SimpleSignature(expr.TDouble)),
    "gqMeanHomRef" ->(22, new SimpleSignature(expr.TDouble)),
    "gqStDevHomRef" ->(23, new SimpleSignature(expr.TDouble)),
    "gqMeanHet" ->(24, new SimpleSignature(expr.TDouble)),
    "gqStDevHet" ->(25, new SimpleSignature(expr.TDouble)),
    "gqMeanHomVar" ->(26, new SimpleSignature(expr.TDouble)),
    "gqStDevHomVar" ->(27, new SimpleSignature(expr.TDouble)),
    "nNonRef" ->(28, new SimpleSignature(expr.TInt)),
    "rTiTv" ->(29, new SimpleSignature(expr.TDouble)),
    "rHetHomVar" ->(30, new SimpleSignature(expr.TDouble)),
    "rDeletionInsertion" ->(31, new SimpleSignature(expr.TDouble))))
}

class SampleQCCombiner extends Serializable {
  var nNotCalled: Int = 0
  var nHomRef: Int = 0
  var nHet: Int = 0
  var nHomVar: Int = 0

  val dpHomRefSC = new StatCounter()
  val dpHetSC = new StatCounter()
  val dpHomVarSC = new StatCounter()

  var nSNP: Int = 0
  var nIns: Int = 0
  var nDel: Int = 0
  var nSingleton: Int = 0
  var nTi: Int = 0
  var nTv: Int = 0

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

  def merge(v: Variant, vIsSingleton: Boolean, g: Genotype): SampleQCCombiner = {
    g.gt match {
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
        if (v.altAllele.isSNP) {
          nSNP += 1
          if (v.altAllele.isTransition)
            nTi += 1
          else {
            assert(v.altAllele.isTransversion)
            nTv += 1
          }
        } else if (v.altAllele.isInsertion)
          nIns += 1
        else if (v.altAllele.isDeletion)
          nDel += 1
        if (vIsSingleton)
          nSingleton += 1
        g.dp.foreach { v =>
          dpHetSC.merge(v)
        }
        g.gq.foreach { v =>
          gqHetSC.merge(v)
        }
      case Some(2) =>
        nHomVar += 1
        if (v.altAllele.isSNP) {
          nSNP += 1
          if (v.altAllele.isTransition)
            nTi += 1
          else {
            assert(v.altAllele.isTransversion)
            nTv += 1
          }
        } else if (v.altAllele.isInsertion)
          nIns += 1
        else if (v.altAllele.isDeletion)
          nDel += 1
        if (vIsSingleton)
          nSingleton += 1
        g.dp.foreach { v =>
          dpHomVarSC.merge(v)
        }
        g.gq.foreach { v =>
          gqHomVarSC.merge(v)
        }
      case None =>
        nNotCalled += 1
      case _ =>
        throw new IllegalArgumentException("Genotype value " + g.gt.get + " must be 0, 1, or 2.")
    }

    this
  }

  def merge(that: SampleQCCombiner): SampleQCCombiner = {
    nNotCalled += that.nNotCalled
    nHomRef += that.nHomRef
    nHet += that.nHet
    nHomVar += that.nHomVar

    nSNP += that.nSNP
    nIns += that.nIns
    nDel += that.nDel
    nSingleton += that.nSingleton
    nTi += that.nTi
    nTv += that.nTv

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

  def emit(sb: mutable.StringBuilder) {

    val nCalled = nHomRef + nHet + nHomVar
    val callRate = divOption(nHomRef + nHet + nHomVar, nHomRef + nHet + nHomVar + nNotCalled)

    sb.tsvAppend(callRate)
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

    sb.append(nSNP)
    sb += '\t'
    sb.append(nIns)
    sb += '\t'
    sb.append(nDel)
    sb += '\t'

    sb.append(nSingleton)
    sb += '\t'

    sb.append(nTi)
    sb += '\t'
    sb.append(nTv)
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

    // nTiTvf
    sb.tsvAppend(divOption(nTi, nTv))
    sb += '\t'

    // rHetHomVar
    sb.tsvAppend(divOption(nHet, nHomVar))
    sb += '\t'

    // rDeletionInsertion
    sb.tsvAppend(divOption(nDel, nIns))
  }

  def asRow: Row = {
    Row.fromSeq(Array(
      divNull(nHomRef + nHet + nHomVar, nHomRef + nHet + nHomVar + nNotCalled),
      nHomRef + nHet + nHomVar,
      nNotCalled,
      nHomRef,
      nHet,
      nHomVar,
      nSNP,
      nIns,
      nDel,
      nSingleton,
      nTi,
      nTv,
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
      divNull(nTi, nTv),
      divNull(nHet, nHomVar),
      divNull(nDel, nIns)))
  }

}

object SampleQC extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-o", aliases = Array("--output"),
      usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "sampleqc"

  def description = "Compute per-sample QC metrics"

  def results(vds: VariantDataset): RDD[(Int, SampleQCCombiner)] = {

    /*
    val singletons = sSingletonVariants(vds)
    val singletonsBc = vds.sparkContext.broadcast(singletons)
    vds
      .aggregateBySampleWithKeys(new SampleQCCombiner)(
        (comb, v, s, g) => comb.merge(v, singletonsBc.value(v), g),
        (comb1, comb2) => comb1.merge(comb2))
        */

    val localSamplesBc = vds.sparkContext.broadcast(vds.localSamples)
    vds
      .rdd
      .mapPartitions[(Int, SampleQCCombiner)] { (it: Iterator[(Variant, Annotation, Iterable[Genotype])]) =>
      val zeroValue = Array.fill[SampleQCCombiner](localSamplesBc.value.length)(new SampleQCCombiner)
      localSamplesBc.value.iterator
        .zip(it.foldLeft(zeroValue) { case (acc, (v, va, gs)) =>
          val vIsSingleton = gs.iterator.existsExactly1(_.isCalledNonRef)
          for ((g, i) <- gs.iterator.zipWithIndex)
            acc(i) = acc(i).merge(v, vIsSingleton, g)
          acc
        }.iterator)
    }.foldByKey(new SampleQCCombiner)((comb1, comb2) => comb1.merge(comb2))
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val output = options.output

    val sampleIdsBc = state.sc.broadcast(vds.sampleIds)

    val r = results(vds)

    if (output != null) {
      hadoopDelete(output, state.hadoopConf, recursive = true)
      r.map { case (s, comb) =>
        val sb = new StringBuilder()
        sb.append(sampleIdsBc.value(s))
        sb += '\t'
        comb.emit(sb)
        sb.result()
      }.writeTable(output, Some("sampleID\t" + SampleQCCombiner.header))
    }
    val rMap = r
      .mapValues(_.asRow)
      .collectAsMap()

    val (newSignatures, fInsert) = vds.saSignatures.insert(List("qc"),
      SampleQCCombiner.signatures)
    val newSampleAnnotations = vds.sampleAnnotations
      .zipWithIndex
      .map { case (sa, s) =>
        fInsert(sa, rMap.get(s))
      }

    state.copy(
      vds = vds.copy(
        sampleAnnotations = newSampleAnnotations,
        saSignatures = newSignatures)
    )
  }
}
