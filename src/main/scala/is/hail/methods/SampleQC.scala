package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import is.hail.utils._
import is.hail.variant.{AltAlleleType, Genotype, Variant, VariantDataset}
import org.apache.spark.util.StatCounter

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
    "rInsertionDeletion"

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
    "rInsertionDeletion" -> TDouble)
}

class SampleQCCombiner(val keepStar: Boolean) extends Serializable {
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

  def merge(v: Variant, acs: Array[Int], g: Genotype): SampleQCCombiner = {

    val gt = Genotype.unboxedGT(g)
    if (gt < 0)
      nNotCalled += 1
    else {

      // FIXME these should probably get merged if gt is uncalled,
      // but don't want to break behavior without a version change
      if (Genotype.unboxedDP(g) >= 0)
        dpSC.merge(Genotype.unboxedDP(g))

      if (Genotype.unboxedGQ(g) >= 0)
        gqSC.merge(Genotype.unboxedGQ(g))

      if (gt == 0) {
        nHomRef += 1
      } else {
        val gtPair = Genotype.gtPair(gt)

        def mergeAllele(ai: Int) {
          if (ai > 0 && (keepStar || !v.altAlleles(ai - 1).isStar)) {

            val altAllele = v.altAlleles(ai - 1)
            altAllele.altAlleleType match {
              case AltAlleleType.SNP =>
                nSNP += 1
                if (altAllele.isTransition)
                  nTi += 1
                else
                  nTv += 1
              case AltAlleleType.Insertion =>
                nIns += 1
              case AltAlleleType.Deletion =>
                nDel += 1
              case _ =>
            }

            if (acs(ai) == 1)
              nSingleton += 1
          }
        }

        mergeAllele(gtPair.j)
        mergeAllele(gtPair.k)

        if (gtPair.j != gtPair.k)
          nHet += 1
        else
          nHomVar += 1
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
      divNull(nIns, nDel))
}

object SampleQC {
  def results(vds: VariantDataset, keepStar: Boolean): Map[Annotation, SampleQCCombiner] = {
    val depth = treeAggDepth(vds.hc, vds.nPartitions)
    vds.sampleIds.iterator
      .zip(
        vds
          .rdd
          .treeAggregate(Array.fill[SampleQCCombiner](vds.nSamples)(new SampleQCCombiner(keepStar)))({ case (acc, (v, (va, gs))) =>

            val acs = Array.fill(v.nAlleles)(0)
            gs.foreach { g =>
              if (Genotype.unboxedGT(g) >= 0) {
                val gtPair = Genotype.gtPair(Genotype.unboxedGT(g))
                acs(gtPair.j) += 1
                acs(gtPair.k) += 1
              }
            }

            var i = 0
            gs.foreach { g =>
              acc(i).merge(v, acs, g)
              i += 1
            }

            acc
          }, { case (comb1, comb2) =>
            for (i <- comb1.indices)
              comb1(i).merge(comb2(i))
            comb1
          }, depth)
          .iterator)
      .toMap
  }

  def apply(vds: VariantDataset, root: String, keepStar: Boolean): VariantDataset = {

    val r = results(vds, keepStar)
    vds.annotateSamples(SampleQCCombiner.signature,
      Parser.parseAnnotationRoot(root, Annotation.SAMPLE_HEAD), { (x: Annotation) =>
        r.get(x).map(_.asAnnotation).orNull
      })
  }
}
