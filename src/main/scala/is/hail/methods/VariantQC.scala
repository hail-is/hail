package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import is.hail.sparkextras.OrderedRDD
import is.hail.stats.LeveneHaldane
import is.hail.utils._
import is.hail.variant.{HTSGenotypeView, GenericDataset, Genotype, Variant, VariantDataset}
import org.apache.spark.util.StatCounter

import scala.reflect.classTag

final class VariantQCCombiner {
  var nNotCalled: Int = 0
  var nHomRef: Int = 0
  var nHet: Int = 0
  var nHomVar: Int = 0

  val dpSC = new StatCounter()

  val gqSC: StatCounter = new StatCounter()

  def mergeGT(gt: Int) {
    (gt: @unchecked) match {
      case 0 => nHomRef += 1
      case 1 => nHet += 1
      case 2 => nHomVar += 1
    }
  }

  def skipGT() {
    nNotCalled += 1
  }

  def mergeDP(dp: Int) {
    dpSC.merge(dp)
  }

  def mergeGQ(gq: Int) {
    gqSC.merge(gq)
  }

  def result(): Annotation = {
    val af = {
      val refAlleles = nHomRef * 2 + nHet
      val altAlleles = nHomVar * 2 + nHet
      divNull(altAlleles, refAlleles + altAlleles)
    }

    val nCalled = nHomRef + nHet + nHomVar
    val callrate = divOption(nCalled, nCalled + nNotCalled)
    val ac = nHet + 2 * nHomVar

    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    val rExpectedHetFreq = divNull(LH.getNumericalMean, n)
    val hweP = LH.exactMidP(nAB)

    Annotation(
      divNull(nCalled, nCalled + nNotCalled),
      ac,
      af,
      nCalled,
      nNotCalled,
      nHomRef,
      nHet,
      nHomVar,
      nullIfNot(dpSC.count > 0, dpSC.mean),
      nullIfNot(dpSC.count > 0, dpSC.stdev),
      nullIfNot(gqSC.count > 0, gqSC.mean),
      nullIfNot(gqSC.count > 0, gqSC.stdev),
      nHet + nHomVar,
      divNull(nHet, nHomRef + nHet + nHomVar),
      divNull(nHet, nHomVar),
      rExpectedHetFreq,
      hweP)
  }
}

object VariantQC {
  val signature = TStruct(
    "callRate" -> TFloat64,
    "AC" -> TInt32,
    "AF" -> TFloat64,
    "nCalled" -> TInt32,
    "nNotCalled" -> TInt32,
    "nHomRef" -> TInt32,
    "nHet" -> TInt32,
    "nHomVar" -> TInt32,
    "dpMean" -> TFloat64,
    "dpStDev" -> TFloat64,
    "gqMean" -> TFloat64,
    "gqStDev" -> TFloat64,
    "nNonRef" -> TInt32,
    "rHeterozygosity" -> TFloat64,
    "rHetHomVar" -> TFloat64,
    "rExpectedHetFrequency" -> TFloat64,
    "pHWE" -> TFloat64)

  def apply(vds: GenericDataset, root: String): GenericDataset = {
    val (newVAS, insertQC) = vds.vaSignature.insert(VariantQC.signature,
      Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD))
    val nSamples = vds.nSamples
    val rowSignature = vds.rowSignature
    val rdd = vds.unsafeRowRDD.mapPartitions { it =>
      val view = HTSGenotypeView(rowSignature)
      it.map { r =>
        view.setRegion(r.region, r.offset)

        val comb = new VariantQCCombiner
        var i = 0
        while (i < nSamples) {
          view.setGenotype(i)
          if (view.hasGT)
            comb.mergeGT(view.getGT)
          else
            comb.skipGT()

          if (view.hasDP)
            comb.mergeDP(view.getDP)
          if (view.hasGQ)
            comb.mergeGQ(view.getGQ)

          i += 1
        }

        (r.get(0): Annotation) -> (insertQC(r.get(1), comb.result()) -> r.getAs[Iterable[Annotation]](2))
      }
    }
    val ord = OrderedRDD.apply(rdd, vds.rdd.orderedPartitioner)(vds.rdd.kOk, classTag[(Annotation, Iterable[Annotation])])
    vds.copy[Annotation, Annotation, Annotation](rdd = ord, vaSignature = newVAS)
  }
}
