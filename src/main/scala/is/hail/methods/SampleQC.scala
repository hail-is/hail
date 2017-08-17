package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import is.hail.utils._
import is.hail.variant.{AltAlleleType, GenericDataset, Genotype, HTSGenotypeView, Variant}
import org.apache.spark.util.StatCounter

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
    "nStar\t" +
    "dpMean\tdpStDev\t" +
    "gqMean\tgqStDev\t" +
    "nNonRef\t" +
    "rTiTv\t" +
    "rHetHomVar\t" +
    "rInsertionDeletion"

  val signature = TStruct("callRate" -> TFloat64,
    "nCalled" -> TInt32,
    "nNotCalled" -> TInt32,
    "nHomRef" -> TInt32,
    "nHet" -> TInt32,
    "nHomVar" -> TInt32,
    "nSNP" -> TInt32,
    "nInsertion" -> TInt32,
    "nDeletion" -> TInt32,
    "nSingleton" -> TInt32,
    "nTransition" -> TInt32,
    "nTransversion" -> TInt32,
    "nStar" -> TInt32,
    "dpMean" -> TFloat64,
    "dpStDev" -> TFloat64,
    "gqMean" -> TFloat64,
    "gqStDev" -> TFloat64,
    "nNonRef" -> TInt32,
    "rTiTv" -> TFloat64,
    "rHetHomVar" -> TFloat64,
    "rInsertionDeletion" -> TFloat64)

  val ti = 0
  val tv = 1
  val ins = 2
  val del = 3
  val star = 4

  def alleleIndices(v: Variant): Array[Int] = {
    val alleleTypes = v.altAlleles.map { a =>
      a.altAlleleType match {
        case AltAlleleType.SNP => if (a.isTransition) ti else tv
        case AltAlleleType.Insertion => ins
        case AltAlleleType.Deletion => del
        case _ => -1
      }
    }.toArray

    alleleTypes
  }
}

class SampleQCCombiner extends Serializable {
  var nNotCalled: Int = 0
  var nHomRef: Int = 0
  var nHet: Int = 0
  var nHomVar: Int = 0

  val aCounts: Array[Int] = Array.fill[Int](5)(0)
  var nSingleton: Int = 0

  val dpSC: StatCounter = new StatCounter()

  val gqSC: StatCounter = new StatCounter()

  def merge(aTypes: Array[Int], acs: Array[Int], view: HTSGenotypeView): SampleQCCombiner = {

    if (view.hasGT) {
      val gt = view.getGT
      if (gt == 0)
        nHomRef += 1
      else {
        val gtPair = Genotype.gtPair(gt)
        val j = gtPair.j
        val k = gtPair.k

        def mergeAllele(idx: Int) {
          if (idx > 0) {
            val aType = aTypes(idx - 1)
            aCounts(aType) += 1
            if (acs(idx) == 1)
              nSingleton += 1
          }
        }

        mergeAllele(j)
        mergeAllele(k)

        if (j != k)
          nHet += 1
        else
          nHomVar += 1
      }
    } else nNotCalled += 1

    if (view.hasDP) {
      dpSC.merge(view.getDP)
    }

    if (view.hasGQ)
      gqSC.merge(view.getGQ)

    this
  }

  def merge(that: SampleQCCombiner): SampleQCCombiner = {
    nNotCalled += that.nNotCalled
    nHomRef += that.nHomRef
    nHet += that.nHet
    nHomVar += that.nHomVar

    aCounts(0) += that.aCounts(0)
    aCounts(1) += that.aCounts(1)
    aCounts(2) += that.aCounts(2)
    aCounts(3) += that.aCounts(3)
    aCounts(4) += that.aCounts(4)

    nSingleton += that.nSingleton

    dpSC.merge(that.dpSC)
    gqSC.merge(that.gqSC)

    this
  }

  def asAnnotation: Annotation = {
    val nTi = aCounts(SampleQCCombiner.ti)
    val nTv = aCounts(SampleQCCombiner.tv)
    val nIns = aCounts(SampleQCCombiner.ins)
    val nDel = aCounts(SampleQCCombiner.del)
    val nStar = aCounts(SampleQCCombiner.star)
    Annotation(
      divNull(nHomRef + nHet + nHomVar, nHomRef + nHet + nHomVar + nNotCalled),
      nHomRef + nHet + nHomVar,
      nNotCalled,
      nHomRef,
      nHet,
      nHomVar,
      nTi + nTv,
      nIns,
      nDel,
      nSingleton,
      nTi,
      nTv,
      nStar,
      nullIfNot(dpSC.count > 0, dpSC.mean),
      nullIfNot(dpSC.count > 0, dpSC.stdev),
      nullIfNot(gqSC.count > 0, gqSC.mean),
      nullIfNot(gqSC.count > 0, gqSC.stdev),
      nHet + nHomVar,
      divNull(nTi, nTv),
      divNull(nHet, nHomVar),
      divNull(nIns, nDel))
  }
}

object SampleQC {
  def results(vds: GenericDataset): Map[Annotation, SampleQCCombiner] = {
    val depth = treeAggDepth(vds.hc, vds.nPartitions)
    val rowSignature = vds.rowSignature
    val nSamples = vds.nSamples
    if (vds.rdd.partitions.nonEmpty)
      vds.sampleIds.iterator
        .zip(vds.unsafeRowRDD
          .mapPartitions { it =>
            val view = HTSGenotypeView(rowSignature)
            val acc = Array.fill[SampleQCCombiner](nSamples)(new SampleQCCombiner)

            it.foreach { r =>
              view.setRegion(r.region, r.offset)
              val v = r.getAs[Variant](0)
              val ais = SampleQCCombiner.alleleIndices(v)
              val acs = Array.fill(v.asInstanceOf[Variant].nAlleles)(0)

              // first pass to compute allele counts
              var i = 0
              while (i < nSamples) {
                view.setGenotype(i)

                if (view.hasGT) {
                  val gt = view.getGT
                  val gtPair = Genotype.gtPair(gt)
                  acs(gtPair.j) += 1
                  acs(gtPair.k) += 1
                }

                i += 1
              }

              // second pass to add to sample statistics
              i = 0
              while (i < nSamples) {
                view.setGenotype(i)
                acc(i).merge(ais, acs, view)
                i += 1
              }
            }

            Iterator(acc)
          }.treeReduce({ case (accs1, accs2) =>
          assert(accs1.length == accs2.length)
          var i = 0
          while (i < accs1.length) {
            accs1(i) = accs1(i).merge(accs2(i))
            i += 1
          }
          accs1
        }, depth).iterator).toMap
    else
      vds.sampleIds.iterator.map(s => (s, new SampleQCCombiner)).toMap
  }

  def apply(vds: GenericDataset, root: String): GenericDataset = {
    val r = results(vds)
    vds.annotateSamples(SampleQCCombiner.signature,
      Parser.parseAnnotationRoot(root, Annotation.SAMPLE_HEAD), { (x: Annotation) => r(x).asAnnotation })
  }
}
