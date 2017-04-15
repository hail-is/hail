package is.hail.methods

import breeze.linalg.DenseMatrix
import is.hail.variant.{Genotype, Variant, VariantDataset}
import is.hail.annotations.Annotation

object MinHash {
  def apply(vds: VariantDataset): Array[Int] = {
    val n = vds.nSamples
    def seqOp(l: Array[Int], r: (Variant, (Annotation, Iterable[Genotype]))): Array[Int] = {
      r match {
        case (v, (_, gs)) => {
          val gsItr = gs.hardCallIterator
          var i = 0
          while(i < n) {
            if (gsItr.next() > 0) l(i) = math.min(v.hashCode(),l(i))
            i += 1
          }
          l
        }
      }
    }
    def combOp(l: Array[Int], r: Array[Int]): Array[Int] = {
      var i = 0
      while(i < n) { l(i) = math.min(l(i), r(i)); i += 1 }
      l
    }
    vds.rdd.aggregate(Array.fill[Int](n)(Int.MaxValue))(seqOp, combOp)
  }
  def apply2(vds: VariantDataset): Array[Int] = {
    vds.rdd.map { case (v, (_, gs)) => (v.hashCode(), gs) }
      .aggregate(Array.fill[Int](vds.nSamples)(Int.MaxValue))({ case (a, (g, gs)) =>
        gs.map(_.gt).zip(a).map { case (gt, i) => if (gt.getOrElse(-1) > 0) math.min(g, i) else i }.toArray },
        (l, r) => (l, r).zipped.map(math.min))
  }
}
