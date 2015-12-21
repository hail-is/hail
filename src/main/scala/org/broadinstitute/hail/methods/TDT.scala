package org.broadinstitute.hail.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils.MultiArray2
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.variant.GenotypeType._
import org.broadinstitute.hail.Utils._

object TDT {

  def apply(vds: VariantDataset, trios: Array[CompleteTrio]): RDD[(Variant, (Int, Int))] = {
    require(trios.forall(_.sex.isDefined)) //FIXME: Rather than throw an exception, simply discard sexless trios and issue warning?

    val sampleTrioRoles: Array[List[(Int, Int)]] = Array.fill[List[(Int, Int)]](vds.nSamples)(List())
    trios.zipWithIndex.foreach { case (t, ti) =>
      sampleTrioRoles(t.kid) ::=(ti, 0)
      sampleTrioRoles(t.dad) ::=(ti, 1)
      sampleTrioRoles(t.mom) ::=(ti, 2)
    }

    val sc = vds.sparkContext
    val sampleTrioRolesBc = sc.broadcast(sampleTrioRoles)
    // all trios have defined sex, see require above
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    val zeroVal: MultiArray2[GenotypeType] = MultiArray2.fill(trios.length, 3)(NoCall)

    def seqOp(a: MultiArray2[GenotypeType], s: Int, g: Genotype): MultiArray2[GenotypeType] = {
      sampleTrioRolesBc.value(s).foreach { case (ti, ri) => a.update(ti, ri, g.gtType) }
      a
    }

    def mergeOp(a: MultiArray2[GenotypeType], b: MultiArray2[GenotypeType]): MultiArray2[GenotypeType] = {
      for ((i, j) <- a.indices)
        if (b(i, j) != NoCall)
          a(i, j) = b(i, j)
      a
    }

    def computeTU(gts: IndexedSeq[GenotypeType], inXnonPAR: Boolean, inYnonPAR: Boolean, sex: Sex.Sex): (Int, Int) = {

      val (gtKid, gtDad, gtMom) = (gts(0), gts(1), gts(2))

      val minAlt: Map[GenotypeType, Int] = Map(HomRef -> 0, Het -> 0, HomVar -> 1)
      val maxAlt: Map[GenotypeType, Int] = Map(HomRef -> 0, Het -> 1, HomVar -> 1)
      val numAlt: Map[GenotypeType, Int] = Map(HomRef -> 0, Het -> 1, HomVar -> 2)
      val numAltH: Map[GenotypeType, Int] = Map(HomRef -> 0, HomVar -> 1)

      //FIXME: Should we issue a warning if there are Mendelian or sex chromosome violations?
      if (((!inXnonPAR && !inYnonPAR) || (inXnonPAR && sex == Sex.Female && gtDad != Het)) &&
        minAlt(gtMom) + minAlt(gtDad) <= numAlt(gtKid) && numAlt(gtKid) <= maxAlt(gtMom) + maxAlt(gtDad))
        (numAlt(gtKid) - minAlt(gtMom) - minAlt(gtDad), maxAlt(gtMom) + maxAlt(gtDad) - numAlt(gtKid))
      else if (inXnonPAR && sex == Sex.Male && gtDad != Het && gtKid != Het &&
        minAlt(gtMom) <= numAltH(gtKid) && numAltH(gtKid) <= maxAlt(gtMom))
        (numAltH(gtKid) - minAlt(gtMom), maxAlt(gtMom) - numAltH(gtKid))
      else
        (0, 0)
    }

    vds
      .aggregateByVariantWithKeys(zeroVal)(
        (a, v, s, g) => seqOp(a, s, g),
        mergeOp)
      .map { case (v, a) =>
        (v, a.rows
          .map(row => computeTU(row, v.inXnonPAR, v.inYnonPAR, trioSexBc.value(row.i)))
          .foldLeft((0, 0)) { case ((t1, u1), (t2, u2)) => (t1 + t2, u1 + u2) })
      }
      .cache()
  }

  def write(tdt: RDD[(Variant, (Int, Int))], filename: String) {
    def toLine(r: (Variant, (Int, Int))): String = {
      val (v, (t, u)) = r
      val chisq = 1.0 * (t - u) * (t - u) / (t + u)
      val p = chiSquaredTail(1, chisq)
      v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\t" + t + "\t" + u + "\t" + chisq + "\t" + p
    }
    tdt.map(toLine).writeTable(filename, "CHR\tPOS\tREF\tALT\tT\tU\tCHISQ\tP\n")
  }

}
