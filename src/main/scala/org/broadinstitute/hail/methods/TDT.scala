package org.broadinstitute.hail.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.MultiArray2
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
      case (HomRef, Het,    Het,    Auto)  => (0, 2)
      case (HomRef, HomRef, Het,    Auto)  => (0, 1)
      case (HomRef, Het,    HomRef, Auto)  => (0, 1)
      case (Het,    Het,    Het,    Auto)  => (1, 1)
      case (Het,    HomRef, Het,    Auto)  => (1, 0)
      case (Het,    Het,    HomRef, Auto)  => (1, 0)
      case (Het,    HomVar, Het,    Auto)  => (0, 1)
      case (Het,    Het,    HomVar, Auto)  => (0, 1)
      case (HomVar, Het,    Het,    Auto)  => (2, 0)
      case (HomVar, Het,    HomVar, Auto)  => (1, 0)
      case (HomVar, HomVar, Het,    Auto)  => (1, 0)
      case (HomRef, HomRef, Het,    HemiX) => (0, 1)
      case (HomRef, HomVar, Het,    HemiX) => (0, 1)
      case (HomVar, HomRef, Het,    HemiX) => (1, 0)
      case (HomVar, HomVar, Het,    HemiX) => (1, 0)
      case _                               => (0, 0) // No transmission
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

  def apply(vds: VariantDataset, preTrios: IndexedSeq[CompleteTrio]): RDD[(Variant, TDTResult)] = {

    // Remove trios with an undefined sex.
    val trios = preTrios.filter(_.sex.isDefined)
    val nTrio = trios.length

    val nSamplesDiscarded = preTrios.length - nTrio
    val noCall = Genotype()

    if (nSamplesDiscarded > 0)
      warn(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} discarded from .fam: missing from variant data set.")

    val sampleTrioRoles = mutable.Map.empty[String, List[(Int, Int)]]

    // Sample trio roles is a hash table with Key = sample ID, value = List[(trio index, value indicating whether the
    //                                                                       sample is a child, dad, or mom)].
    // Because a sample can be both a child in one family and a parent in another, the List can contain
    // multiple sets of (trio index, family identity (i.e., kid, dad, mom)).
    trios.zipWithIndex.foreach { case (t, tidx) =>
      sampleTrioRoles += (t.kid -> sampleTrioRoles.getOrElse(t.kid, List.empty[(Int, Int)]).::(tidx, 0))
      sampleTrioRoles += (t.dad -> sampleTrioRoles.getOrElse(t.dad, List.empty[(Int, Int)]).::(tidx, 1))
      sampleTrioRoles += (t.mom -> sampleTrioRoles.getOrElse(t.mom, List.empty[(Int, Int)]).::(tidx, 2))
    }

    val sc = vds.sparkContext
    val sampleTrioRolesBc = sc.broadcast(sampleTrioRoles)
    val triosBc = sc.broadcast(trios)

    // All trios have defined sex, see filter above.
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    // zeroVal is a n x 3 array, where n is the number of trios. Each of the three columns corresponds to kid, dad, mom.
    // This n x 3 array will eventually hold the genotypes of the trios.
    val zeroVal: MultiArray2[GenotypeType] = MultiArray2.fill(nTrio, 3)(NoCall)

    def seqOp(a: MultiArray2[GenotypeType], v: Variant, s: String, g: Genotype): MultiArray2[GenotypeType] = {
      sampleTrioRolesBc.value.get(s).foreach(l => l.foreach { case (tidx, ridx) =>
        // Ignore Fathers who are heterozygous outside of the pseudoautosomal region of the X chromosome.
        if (v.contig == "X" && !v.inXPar && ridx == 1 && g.isHet)
          a.update(tidx, ridx, GenotypeType.NoCall)
        else
          a.update(tidx, ridx, g.gtType)
      })
      a
    }

    def mergeOp(a: MultiArray2[GenotypeType], b: MultiArray2[GenotypeType]): MultiArray2[GenotypeType] = {
      for ((i, j) <- a.indices)
        if (b(i, j) != NoCall)
          a(i, j) = b(i, j)
      a
    }

    val combOp: (Array[(Int, Int)], Array[(Int, Int)]) => Array[(Int, Int)] = {
      case (a1, a2) =>
        a1.zip(a2)
          .map { case ((t1, u1), (t2, u2)) => (t1 + t2, u1 + u2) }
    }

    val transmissionMatrix = vds
      .filterVariants((v, va, gs) => (v.contig != "Y") && (v.contig != "MT"))
      .aggregateByVariantWithKeys(zeroVal)(
        (a, v, s, g) => seqOp(a, v, s, g),
        mergeOp)
      .map { case (v, ma) =>
        (v, ma.rowIndices.map { i => getTransmission(ma(i, 0), ma(i, 1), ma(i, 2), v.copyState(trioSexBc.value(i)))}.toArray)
        }

    // per variant:
    val perVariantRDD = transmissionMatrix.map { case (v, transmissions) =>
      val (totalTrans, totalUntrans) = transmissions.fold((0, 0)){ case ((t1,u1), (t2,u2)) => (t1 + t2, u1 + u2)}
      (v, TDTResult(totalTrans, totalUntrans, calcTDTstat(totalTrans, totalUntrans)))
    }

    // per trio:
//    val perTrio = transmissionMatrix
//      .map(_._2)
//     .treeReduce({ case (a1, a2) =>
//     Array.tabulate[(Int, Int)](nTrio){ i =>
//        val (t1, u1) = a1(i)
//        val (t2, u2) = a2(i)
//       (t1 + t2, u1 + u2)
//     }})
//
//    val sampleMap = trios.zipWithIndex
//      .map { case (t, i) => (t.kid, }

    perVariantRDD
  }
}
