package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats.chiSquaredTail
import is.hail.utils._
import is.hail.variant.CopyState._
import is.hail.variant.GenotypeType._
import is.hail.variant._

import scala.collection.mutable

case class TDTResult(nTransmitted: Int, nUntransmitted: Int, chi2: Double, pval: Double) {
  def toAnnotation: Annotation = Annotation(nTransmitted, nUntransmitted, chi2, pval)
}

object TDT {

  def schema: TStruct = TStruct("nTransmitted" -> TInt32, "nUntransmitted" -> TInt32, "chi2" -> TFloat64, "pval" -> TFloat64)

  def getTransmission(kid: GenotypeType, dad: GenotypeType, mom: GenotypeType, copyState: CopyState): (Int, Int) = {
    (kid, dad, mom, copyState) match {
      // (kid's genotype, dad's genotype, mom's genotype)
      case (HomRef, Het, Het, Auto) => (0, 2)
      case (HomRef, HomRef, Het, Auto) => (0, 1)
      case (HomRef, Het, HomRef, Auto) => (0, 1)
      case (Het, Het, Het, Auto) => (1, 1)
      case (Het, HomRef, Het, Auto) => (1, 0)
      case (Het, Het, HomRef, Auto) => (1, 0)
      case (Het, HomVar, Het, Auto) => (0, 1)
      case (Het, Het, HomVar, Auto) => (0, 1)
      case (HomVar, Het, Het, Auto) => (2, 0)
      case (HomVar, Het, HomVar, Auto) => (1, 0)
      case (HomVar, HomVar, Het, Auto) => (1, 0)
      case (HomRef, HomRef, Het, HemiX) => (0, 1)
      case (HomRef, HomVar, Het, HemiX) => (0, 1)
      case (HomVar, HomRef, Het, HemiX) => (1, 0)
      case (HomVar, HomVar, Het, HemiX) => (1, 0)
      case _ => (0, 0) // No transmission
    }
  }


  def calcTDTstat(t: Double, u: Double): Double = {
    // The TDT uses a McNemar based statistic (which is a 1 df Chi-Square).
    //        (t - u)^2
    //      -------------
    //        (t + u)

    if ((t + u) != 0)
      (t - u) * (t - u) / (t + u)
    else
     0.0
  }

  def apply(vds: VariantDataset, preTrios: IndexedSeq[CompleteTrio], path: List[String]): VariantDataset = {
    vds.requireUniqueSamples("tdt")

    // Remove trios with an undefined sex.
    val trios = preTrios.filter(_.sex.isDefined)
    val nTrio = trios.length

    val nSamplesDiscarded = preTrios.length - nTrio

    info(s"using $nTrio trios for transmission analysis")
    if (nSamplesDiscarded > 0)
      warn(s"$nSamplesDiscarded ${ plural(nSamplesDiscarded, "sample") } discarded from .fam: missing from variant data set.")

    val sampleTrioRolesMap = mutable.Map.empty[String, List[(Int, Int)]]

    trios.zipWithIndex.foreach { case (t, tidx) =>
      sampleTrioRolesMap += (t.kid -> ((tidx, 0) :: sampleTrioRolesMap.getOrElse(t.kid, Nil)))
      sampleTrioRolesMap += (t.knownDad -> ((tidx, 1) :: sampleTrioRolesMap.getOrElse(t.knownDad, Nil)))
      sampleTrioRolesMap += (t.knownMom -> ((tidx, 2) :: sampleTrioRolesMap.getOrElse(t.knownMom, Nil)))
    }

    val sampleTrioRolesArray = vds.stringSampleIds.map(s => sampleTrioRolesMap.getOrElse(s, Nil))

    val sc = vds.sparkContext
    val sampleTrioRolesBc = vds.sparkContext.broadcast(sampleTrioRolesArray)

    // All trios have defined sex, see filter above.
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    val (newVA, inserter) = vds.insertVA(schema, path)

    vds.copy(rdd = vds.rdd.mapPartitions({ it =>

      val arr = MultiArray2.fill(nTrio, 3)(NoCall)
      it.map { case (v, (va, gs)) =>
        if (v.isMitochondrial || v.inYNonPar)
          (v, (inserter(va, null), gs))
        else {
          gs.iterator.zipWithIndex.foreach { case (g, i) =>
            sampleTrioRolesBc.value(i).foreach { case (tIdx, rIdx) =>
              if (v.inXNonPar && rIdx == 1 && Genotype.isHet(g))
                arr.update(tIdx, rIdx, GenotypeType.NoCall)
              else
                arr.update(tIdx, rIdx, Genotype.gtType(g))
            }
          }

          var t = 0
          var u = 0
          var i = 0
          while (i < arr.n1) {
            val (nt, nu) = getTransmission(arr(i, 0), arr(i, 1), arr(i, 2), v.copyState(trioSexBc.value(i)))
            t += nt
            u += nu
            i += 1
          }

          val chi2 = calcTDTstat(t, u)
          val pval = chiSquaredTail(1.0, chi2)
          val tdtAnnotation = Annotation(t, u, chi2, pval)
          (v, (inserter(va, tdtAnnotation), gs))
        }
      }
    }, preservesPartitioning = true).asOrderedRDD).copy(vaSignature = newVA)
  }
}
