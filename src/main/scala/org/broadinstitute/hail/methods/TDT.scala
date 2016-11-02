package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.utils.{MultiArray2, _}
import org.broadinstitute.hail.variant.CopyState._
import org.broadinstitute.hail.variant.GenotypeType._
import org.broadinstitute.hail.variant._

import scala.collection.mutable

case class TDTResult(nTransmitted: Int, nUntransmitted: Int, chiSquare: Double) {
  def toAnnotation: Annotation = Annotation(nTransmitted, nUntransmitted, chiSquare)
}

object TDT {

  def schema: TStruct = TStruct("nTransmitted" -> TInt, "nUntransmitted" -> TInt, "chiSquare" -> TDouble)

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


  def calcTDTstat(transmitted: Int, untransmitted: Int): Double = {
    // The TDT uses a McNemar based statistic (which is a 1 df Chi-Square).
    //        (T - U)^2
    //      -------------
    //        (T + U)

    var chiSquare = 0.0

    if ((transmitted + untransmitted) != 0)
      chiSquare = scala.math.pow(transmitted - untransmitted, 2.0) / (transmitted + untransmitted)

    chiSquare
  }

  def apply(vds: VariantDataset, preTrios: IndexedSeq[CompleteTrio], path: List[String]): VariantDataset = {

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
      sampleTrioRolesMap += (t.dad -> ((tidx, 1) :: sampleTrioRolesMap.getOrElse(t.dad, Nil)))
      sampleTrioRolesMap += (t.mom -> ((tidx, 2) :: sampleTrioRolesMap.getOrElse(t.mom, Nil)))
    }

    val sampleTrioRolesArray = vds.sampleIds.map(sampleTrioRolesMap.getOrElse(_, Nil))

    val sc = vds.sparkContext
    val sampleTrioRolesBc = vds.sparkContext.broadcast(sampleTrioRolesArray)

    // All trios have defined sex, see filter above.
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    val (newVA, inserter) = vds.insertVA(schema, path)

    vds.copy(rdd = vds.rdd.mapPartitions({ it =>

      val arr = MultiArray2.fill(nTrio, 3)(NoCall)
      it.map { case (v, (va, gs)) =>
        if (v.isMitochondrial || v.inYNonPar)
          (v, (inserter(va, None), gs))
        else {
          gs.iterator.zipWithIndex.foreach { case (g, i) =>
            sampleTrioRolesBc.value(i).foreach { case (tIdx, rIdx) =>
              if (v.inXNonPar && rIdx == 1 && g.isHet)
                arr.update(tIdx, rIdx, GenotypeType.NoCall)
              else
                arr.update(tIdx, rIdx, g.gtType)
            }
          }

          var nTrans = 0
          var nUntrans = 0
          var i = 0
          while (i < arr.n1) {
            val (nt, nu) = getTransmission(arr(i, 0), arr(i, 1), arr(i, 2), v.copyState(trioSexBc.value(i)))
            nTrans += nt
            nUntrans += nu
            i += 1
          }

          val tdtAnnotation = Annotation(nTrans, nUntrans, calcTDTstat(nTrans, nUntrans))
          (v, (inserter(va, Some(tdtAnnotation)), gs))
        }
      }
    }, preservesPartitioning = true).asOrderedRDD).copy(vaSignature = newVA)
  }
}
