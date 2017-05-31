package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import is.hail.stats.LeveneHaldane
import is.hail.utils._
import is.hail.variant.{Genotype, Variant, VariantDataset}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter

import scala.collection.mutable


object VariantQCCombiner {
  val header =
    "callRate\t" +
      "AC\t" +
      "AF\t" +
      "nCalled\t" +
      "nNotCalled\t" +
      "nHomRef\t" +
      "nHet\t" +
      "nHomVar\t" +
      "dpMean\tdpStDev\t" +
      "gqMean\tgqStDev\t" +
      "nNonRef\t" +
      "rHeterozygosity\t" +
      "rHetHomVar\t" +
      "rExpectedHetFrequency\tpHWE"

  val signature = TStruct(
    "callRate" -> TDouble,
    "AC" -> TInt,
    "AF" -> TDouble,
    "nCalled" -> TInt,
    "nNotCalled" -> TInt,
    "nHomRef" -> TInt,
    "nHet" -> TInt,
    "nHomVar" -> TInt,
    "dpMean" -> TDouble,
    "dpStDev" -> TDouble,
    "gqMean" -> TDouble,
    "gqStDev" -> TDouble,
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

  val dpSC = new StatCounter()

  val gqSC: StatCounter = new StatCounter()

  // FIXME per-genotype

  def merge(g: Genotype): VariantQCCombiner = {
    (Genotype.gt(g): @unchecked) match {
      case Some(0) =>
        nHomRef += 1
      case Some(1) =>
        nHet += 1
      case Some(2) =>
        nHomVar += 1
      case None =>
        nNotCalled += 1
    }

    if (Genotype.isCalled(g)) {
      Genotype.dp(g).foreach { v =>
        dpSC.merge(v)
      }
      Genotype.gq(g).foreach { v =>
        gqSC.merge(v)
      }
    }

    this
  }

  def merge(that: VariantQCCombiner): VariantQCCombiner = {
    nNotCalled += that.nNotCalled
    nHomRef += that.nHomRef
    nHet += that.nHet
    nHomVar += that.nHomVar

    dpSC.merge(that.dpSC)

    gqSC.merge(that.gqSC)

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
    val ac = nHet + 2 * nHomVar

    sb.tsvAppend(callRate)
    sb += '\t'
    sb.append(ac)
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

    emitSC(sb, gqSC)
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
    val af = {
      val refAlleles = nHomRef * 2 + nHet
      val altAlleles = nHomVar * 2 + nHet
      divOption(altAlleles, refAlleles + altAlleles)
    }

    val nCalled = nHomRef + nHet + nHomVar
    val hwe = HWEStats
    val callrate = divOption(nCalled, nCalled + nNotCalled)
    val ac = nHet + 2 * nHomVar

    Annotation(
      divNull(nCalled, nCalled + nNotCalled),
      ac,
      af.getOrElse(null),
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
      hwe._1.getOrElse(null),
      hwe._2)
  }
}

object VariantQC {
  def results(vds: VariantDataset): RDD[(Variant, VariantQCCombiner)] =
    vds
      .aggregateByVariant(new VariantQCCombiner)((comb, g) => comb.merge(g),
        (comb1, comb2) => comb1.merge(comb2))

  def apply(vds: VariantDataset, root: String): VariantDataset = {
    val (newVAS, insertQC) = vds.vaSignature.insert(VariantQCCombiner.signature,
      Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD))
    vds.mapAnnotationsWithAggregate(new VariantQCCombiner, newVAS)((comb, v, va, s, sa, g) => comb.merge(g),
      (comb1, comb2) => comb1.merge(comb2),
      (va, comb) => insertQC(va, comb.asAnnotation))
  }
}
