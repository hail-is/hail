package org.broadinstitute.hail.methods

import breeze.linalg.max
import org.broadinstitute.hail.variant.{Genotype, VariantSampleMatrix, Variant}

object LinkageDisiquilibrium {
  def v(g: Genotype): (Int, Int) = {
    val o = g.call.map(_.gt)
    if (o.isEmpty) (0, 0) else (o.get, 1)
  }

  def E(A: Iterable[(Int, Int)]): Float = {
    val sums = A.reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    sums._1.toFloat / sums._2
  }

  def r2(A: Iterable[Genotype], B: Iterable[Genotype]): Double = {
    val ea =  E(A.map(a => v(a)))
    val eb = E(B.map(a => v(a)))
    val eab = E(A.zip(B).map(a => (v(a._1)._1 * v(a._2)._1, v(a._1)._2 * v(a._2)._2)))
    Math.pow(eab - ea * eb, 2.0) / (ea * eb)
  }

  def Prune(variants: Iterable[(Variant, Iterable[Genotype])],
            maxld: Double, minfreq: Double): Iterable[(Variant, Iterable[Genotype])] = {
    PruneWindow(
      variants.filter(V => (V._2.map(a => v(a)._2).sum.toFloat / max(1, V._2.size)) > minfreq),
      maxld)
  }

  def PruneWindow(variants: Iterable[(Variant, Iterable[Genotype])],
                  maxld: Double): Iterable[(Variant, Iterable[Genotype])] = {
    if (variants.isEmpty)
      List()
    else {
      val keeper = variants.head
      List(keeper) ++
        PruneWindow(variants.drop(1).filter(v => r2(keeper._2, v._2) < maxld), maxld)
    }
  }
}
