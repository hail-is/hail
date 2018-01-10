package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr.Parser
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.{AltAlleleType, Genotype, HTSGenotypeView, MatrixTable, Variant}
import org.apache.spark.util.StatCounter

object SampleQCCombiner {
  val signature = TStruct("callRate" -> TFloat64(),
    "nCalled" -> TInt64(),
    "nNotCalled" -> TInt64(),
    "nHomRef" -> TInt64(),
    "nHet" -> TInt64(),
    "nHomVar" -> TInt64(),
    "nSNP" -> TInt64(),
    "nInsertion" -> TInt64(),
    "nDeletion" -> TInt64(),
    "nSingleton" -> TInt64(),
    "nTransition" -> TInt64(),
    "nTransversion" -> TInt64(),
    "nStar" -> TInt64(),
    "dpMean" -> TFloat64(),
    "dpStDev" -> TFloat64(),
    "gqMean" -> TFloat64(),
    "gqStDev" -> TFloat64(),
    "nNonRef" -> TInt64(),
    "rTiTv" -> TFloat64(),
    "rHetHomVar" -> TFloat64(),
    "rInsertionDeletion" -> TFloat64())

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
        case AltAlleleType.Star => star
        case _ => -1
      }
    }.toArray

    alleleTypes
  }
}

class SampleQCCombiner extends Serializable {
  var nNotCalled: Long = 0
  var nHomRef: Long = 0
  var nHet: Long = 0
  var nHomVar: Long = 0

  val aCounts: Array[Long] = Array.fill[Long](5)(0L)
  var nSingleton: Long = 0

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
            if (aType >= 0) // if the type is one we are tracking
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

    var i = 0
    while (i <= SampleQCCombiner.star) {
      aCounts(i) += that.aCounts(i)
      i += 1
    }

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
  def results(vsm: MatrixTable): Array[SampleQCCombiner] = {
    val depth = treeAggDepth(vsm.hc, vsm.nPartitions)
    val rowType = vsm.rowType
    val nSamples = vsm.nSamples
    if (vsm.rdd2.partitions.nonEmpty)
      vsm.rdd2
          .mapPartitions { it =>
            val view = HTSGenotypeView(rowType)
            val acc = Array.fill[SampleQCCombiner](nSamples)(new SampleQCCombiner)

            it.foreach { rv =>
              view.setRegion(rv)
              val v = Variant.fromRegionValue(rv.region,
                rowType.loadField(rv, 1))
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

            Iterator.single(acc)
          }.treeReduce({ case (accs1, accs2) =>
          assert(accs1.length == accs2.length)
          var i = 0
          while (i < accs1.length) {
            accs1(i) = accs1(i).merge(accs2(i))
            i += 1
          }
          accs1
        }, depth)
    else
      Array.fill(nSamples)(new SampleQCCombiner)
  }

  def apply(vsm: MatrixTable, root: String = "sa.qc"): MatrixTable = {
    vsm.requireRowKeyVariant("sample_qc")
    val r = results(vsm).map(_.asAnnotation)
    vsm.annotateSamples(SampleQCCombiner.signature, Parser.parseAnnotationRoot(root, Annotation.SAMPLE_HEAD), r)
  }
}
