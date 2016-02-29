package org.broadinstitute.hail.driver

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.methods._
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
    "callRate" -> new SimpleSignature(SignatureType.Double, 0),
    "nCalled" -> new SimpleSignature(SignatureType.Int, 1),
    "nNotCalled" -> new SimpleSignature(SignatureType.Int, 2),
    "nHomRef" -> new SimpleSignature(SignatureType.Int, 3),
    "nHet" -> new SimpleSignature(SignatureType.Int, 4),
    "nHomVar" -> new SimpleSignature(SignatureType.Int, 5),
    "nSNP" -> new SimpleSignature(SignatureType.Int, 6),
    "nInsertion" -> new SimpleSignature(SignatureType.Int, 7),
    "nDeletion" -> new SimpleSignature(SignatureType.Int, 8),
    "nSingleton" -> new SimpleSignature(SignatureType.Int, 9),
    "nTransition" -> new SimpleSignature(SignatureType.Int, 10),
    "nTransversion" -> new SimpleSignature(SignatureType.Int, 11),
    "dpMean" -> new SimpleSignature(SignatureType.Double, 12),
    "dpStDev" -> new SimpleSignature(SignatureType.Double, 13),
    "dpMeanHomRef" -> new SimpleSignature(SignatureType.Double, 14),
    "dpStDevHomRef" -> new SimpleSignature(SignatureType.Double, 15),
    "dpMeanHet" -> new SimpleSignature(SignatureType.Double, 16),
    "dpStDevHet" -> new SimpleSignature(SignatureType.Double, 17),
    "dpMeanHomVar" -> new SimpleSignature(SignatureType.Double, 18),
    "dpStDevHomVar" -> new SimpleSignature(SignatureType.Double, 19),
    "gqMean" -> new SimpleSignature(SignatureType.Double, 20),
    "gqStDev" -> new SimpleSignature(SignatureType.Double, 21),
    "gqMeanHomRef" -> new SimpleSignature(SignatureType.Double, 22),
    "gqStDevHomRef" -> new SimpleSignature(SignatureType.Double, 23),
    "gqMeanHet" -> new SimpleSignature(SignatureType.Double, 24),
    "gqStDevHet" -> new SimpleSignature(SignatureType.Double, 25),
    "gqMeanHomVar" -> new SimpleSignature(SignatureType.Double, 26),
    "gqStDevHomVar" -> new SimpleSignature(SignatureType.Double, 27),
    "nNonRef" -> new SimpleSignature(SignatureType.Int, 28),
    "rTiTv" -> new SimpleSignature(SignatureType.Double, 29),
    "rHetHomVar" -> new SimpleSignature(SignatureType.Double, 30),
    "rDeletionInsertion" -> new SimpleSignature(SignatureType.Double, 31)))
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
      .mapPartitions[(Int, SampleQCCombiner)] { (it: Iterator[(Variant, AnnotationData, Iterable[Genotype])]) =>
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

    val (newSignatures, fInsert) = AnnotationData.insertSignature(vds.metadata.sampleAnnotationSignatures,
      SampleQCCombiner.signatures, Array("qc"))
    val newSampleAnnotations = vds.metadata.sampleAnnotations
      .zipWithIndex
      .map { case (sa, s) =>
        rMap.get(s) match {
          case Some(sa2) => fInsert(sa, sa2)
          case None => sa
        }
      }

        state.copy(
          vds = vds.copy(
            metadata = vds.metadata.copy(sampleAnnotations = newSampleAnnotations,
              sampleAnnotationSignatures = newSignatures)
          ))
  }
}
