package is.hail.methods

import is.hail.annotations._
import is.hail.expr.{TStruct, _}
import is.hail.stats.LeveneHaldane
import is.hail.variant.{HTSGenotypeView, VariantSampleMatrix}
import org.apache.spark.util.StatCounter

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

  def add(rvb: RegionValueBuilder) {
    val nCalled = nHomRef + nHet + nHomVar
    val n = nCalled + nNotCalled

    val ac = nHet + 2 * nHomVar

    val nAB = nHet
    val nA = nAB + 2 * math.min(nHomRef, nHomVar)

    val LH = LeveneHaldane(nCalled, nA)
    val hweP = LH.exactMidP(nAB)

    rvb.startStruct() // qc

    // callRate
    if (n != 0)
      rvb.addDouble(nCalled.toDouble / n)
    else
      rvb.setMissing()

    rvb.addInt(ac)

    // af
    if (nCalled != 0)
      rvb.addDouble(ac.toDouble / (2 * nCalled))
    else
      rvb.setMissing()

    rvb.addInt(nCalled)
    rvb.addInt(nNotCalled)
    rvb.addInt(nHomRef)
    rvb.addInt(nHet)
    rvb.addInt(nHomVar)

    if (dpSC.count > 0) {
      rvb.addDouble(dpSC.mean)
      rvb.addDouble(dpSC.stdev)
    } else {
      rvb.setMissing()
      rvb.setMissing()
    }

    if (gqSC.count > 0) {
      rvb.addDouble(gqSC.mean)
      rvb.addDouble(gqSC.stdev)
    } else {
      rvb.setMissing()
      rvb.setMissing()
    }

    rvb.addInt(nHet + nHomVar)

    if (nCalled != 0)
      rvb.addDouble(nHet.toDouble / nCalled)
    else
      rvb.setMissing()

    if (nHomVar != 0)
      rvb.addDouble(nHet.toDouble / nHomVar)
    else
      rvb.setMissing()

    // rExpectedHetFreq
    if (nCalled != 0)
      rvb.addDouble(LH.getNumericalMean / nCalled)
    else
      rvb.setMissing()

    rvb.addDouble(hweP)
    rvb.endStruct()
  }
}

object VariantQC {
  val signature = TStruct(
    "callRate" -> TFloat64(),
    "AC" -> TInt32(),
    "AF" -> TFloat64(),
    "nCalled" -> TInt32(),
    "nNotCalled" -> TInt32(),
    "nHomRef" -> TInt32(),
    "nHet" -> TInt32(),
    "nHomVar" -> TInt32(),
    "dpMean" -> TFloat64(),
    "dpStDev" -> TFloat64(),
    "gqMean" -> TFloat64(),
    "gqStDev" -> TFloat64(),
    "nNonRef" -> TInt32(),
    "rHeterozygosity" -> TFloat64(),
    "rHetHomVar" -> TFloat64(),
    "rExpectedHetFrequency" -> TFloat64(),
    "pHWE" -> TFloat64())

  def apply(vsm: VariantSampleMatrix, root: String): VariantSampleMatrix = {
    val localNSamples = vsm.nSamples
    val localRowType = vsm.rowType

    vsm.insertIntoRow(() => HTSGenotypeView(localRowType))(VariantQC.signature,
      "va" :: Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD), { (view, rv, rvb) =>
        view.setRegion(rv.region, rv.offset)
        val comb = new VariantQCCombiner
        var i = 0
        while (i < localNSamples) {
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

        comb.add(rvb)
      })
  }
}
