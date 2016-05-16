package org.broadinstitute.hail.driver

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
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
    "gqMean\tgqStDev\t" +
    "nNonRef\t" +
    "rTiTv\t" +
    "rHetHomVar\t" +
    "rDeletionInsertion"

  val signature = TStruct("callRate" -> TDouble,
    "nCalled" -> TInt,
    "nNotCalled" -> TInt,
    "nHomRef" -> TInt,
    "nHet" -> TInt,
    "nHomVar" -> TInt,
    "nSNP" -> TInt,
    "nInsertion" -> TInt,
    "nDeletion" -> TInt,
    "nSingleton" -> TInt,
    "nTransition" -> TInt,
    "nTransversion" -> TInt,
    "dpMean" -> TDouble,
    "dpStDev" -> TDouble,
    "gqMean" -> TDouble,
    "gqStDev" -> TDouble,
    "nNonRef" -> TInt,
    "rTiTv" -> TDouble,
    "rHetHomVar" -> TDouble,
    "rDeletionInsertion" -> TDouble)
}

class SampleQCCombiner extends Serializable {
  var nNotCalled: Int = 0
  var nHomRef: Int = 0
  var nHet: Int = 0
  var nHomVar: Int = 0

  var nSNP: Int = 0
  var nIns: Int = 0
  var nDel: Int = 0
  var nSingleton: Int = 0
  var nTi: Int = 0
  var nTv: Int = 0

  val dpSC: StatCounter = new StatCounter()

  val gqSC: StatCounter = new StatCounter()

  // FIXME per-genotype

  def merge(v: Variant, vIsSingleton: Boolean, g: Genotype): SampleQCCombiner = {
    g.gt match {
      case Some(0) =>
        nHomRef += 1
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
      case None =>
        nNotCalled += 1
      case _ =>
        throw new IllegalArgumentException("Genotype value " + g.gt.get + " must be 0, 1, or 2.")
    }

    if (g.isCalled) {
      g.dp.foreach { v =>
        dpSC.merge(v)
      }
      g.gq.foreach { v =>
        gqSC.merge(v)
      }
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

    dpSC.merge(that.dpSC)
    gqSC.merge(that.gqSC)

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

    emitSC(sb, gqSC)
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

  def asAnnotation: Annotation =
    Annotation(
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
      nullIfNot(gqSC.count > 0, gqSC.mean),
      nullIfNot(gqSC.count > 0, gqSC.stdev),
      nHet + nHomVar,
      divNull(nTi, nTv),
      divNull(nHet, nHomVar),
      divNull(nDel, nIns))
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

  def supportsMultiallelic = false

  def requiresVDS = true

  def results(vds: VariantDataset): Map[String, SampleQCCombiner] = {
    vds.sampleIds.iterator
      .zip(
        vds
          .rdd
          .treeAggregate(Array.fill[SampleQCCombiner](vds.nSamples)(new SampleQCCombiner))({ case (acc, (v, va, gs)) =>
            val vIsSingleton = gs.iterator.existsExactly1(_.isCalledNonRef)
            for ((g, i) <- gs.iterator.zipWithIndex)
              acc(i).merge(v, vIsSingleton, g)
            acc
          }, { case (comb1, comb2) =>
            for (i <- comb1.indices)
              comb1(i).merge(comb2(i))
            comb1
          })
          .iterator)
      .toMap
  }

  /*
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
  } */

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val output = options.output

    val r = results(vds)

    if (output != null) {
      val sb = new StringBuilder()
      hadoopDelete(output, state.hadoopConf, recursive = true)
      writeTable(output, state.hadoopConf,
        r.map { case (s, comb) =>
          sb.clear()
          sb.append(s)
          sb += '\t'
          comb.emit(sb)
          sb.result()
        }, Some("sampleID\t" + SampleQCCombiner.header))
    }

    val (newSAS, insertQC) = vds.saSignature.insert(SampleQCCombiner.signature, "qc")
    val newSampleAnnotations = vds.sampleIdsAndAnnotations
      .map { case (s, sa) =>
        insertQC(sa, r.get(s).map(_.asAnnotation))
      }

    state.copy(
      vds = vds.copy(
        sampleAnnotations = newSampleAnnotations,
        saSignature = newSAS)
    )
  }
}
