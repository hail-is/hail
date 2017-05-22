package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import is.hail.utils._
import is.hail.variant.{Genotype, Variant, VariantDataset}
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

  def merge(v: Variant, ACs: Array[Int], g: Genotype): SampleQCCombiner = {

    g.gt match {

      case Some(0) =>
        nHomRef += 1

      case Some(gt) =>
        val nonRefAlleleIndices = Genotype.gtPair(gt).alleleIndices.filter(i => i > 0 && (keepStar || !v.altAlleles(i - 1).isStar))

        if (!nonRefAlleleIndices.isEmpty) {
          nonRefAlleleIndices.foreach({
            ai =>
              val altAllele = v.altAlleles(ai - 1)
              if (altAllele.isSNP) {
                nSNP += 1
                if (altAllele.isTransition)
                  nTi += 1
                else {
                  assert(altAllele.isTransversion)
                  nTv += 1
                }
              } else if (altAllele.isInsertion)
                nIns += 1
              else if (altAllele.isDeletion)
                nDel += 1

              if (ACs(ai - 1) == 1)
                nSingleton += 1
          })

          if (nonRefAlleleIndices.length == 1 || nonRefAlleleIndices(0) != nonRefAlleleIndices(1))
            nHet += 1
          else
            nHomVar += 1
        }
      case None =>
        nNotCalled += 1

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

    // rInsertionDeletion
    sb.tsvAppend(divOption(nIns, nDel))
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

            val ACs = gs.foldLeft(Array.fill(v.nAltAlleles)(0))({
              case (acc, g) =>
                g.gt
                  .filter(_ > 0)
                  .foreach(call => Genotype.gtPair(call).alleleIndices.filter(_ > 0).foreach(x => acc(x - 1) += 1)
                  )
                acc
            })
            for ((g, i) <- gs.iterator.zipWithIndex)
              acc(i).merge(v, ACs, g)
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
