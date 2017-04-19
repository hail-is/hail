package is.hail.methods

import breeze.linalg.{DenseMatrix, DenseVector, min}
import is.hail.variant.{Genotype, Variant, VariantDataset}
import is.hail.annotations.Annotation
import scala.util.hashing.{MurmurHash3 => MH3}

object MinHash {
  def apply(vds: VariantDataset): Array[Int] = {
    val n = vds.nSamples
    def seqOp(l: Array[Int], r: (Variant, (Annotation, Iterable[Genotype]))): Array[Int] = {
      r match {
        case (v, (_, gs)) =>
          val gsItr = gs.hardCallGenotypeIterator
          var i = 0
          while(i < n) {
            if (gsItr.next() > 0) l(i) = math.min(v.hashCode(),l(i))
            i += 1
          }
          l
      }
    }
    def combOp(l: Array[Int], r: Array[Int]): Array[Int] = {
      var i = 0
      while(i < n) { l(i) = math.min(l(i), r(i)); i += 1 }
      l
    }
    vds.rdd.aggregate(Array.fill[Int](n)(Int.MaxValue))(seqOp, combOp)
  }

  // returns a k by vds.nSamples matrix
  def kMinHash(vds: VariantDataset, k: Int): DenseMatrix[Int] = {
    val n = vds.nSamples
    def seqOp(l: DenseMatrix[Int], r: (Variant, (Annotation, Iterable[Genotype]))): DenseMatrix[Int] = r match {
      case (v, (_, gs)) =>
        val gsItr = gs.hardCallGenotypeIterator
        val hashes = hashChain(k, v.hashCode)
        var i = 0
        while(i < n) {
          //TOTO: Write a min UFunc that modifies the left arg in place
          if (gsItr.next() > 0) l(::,i) := min(l(::,i), hashes)
          i += 1
        }
        l
    }
    val zeroValue = DenseMatrix.fill[Int](k, n)(Int.MaxValue)
    vds.rdd.aggregate(zeroValue)(seqOp, min(_,_))
  }

  def apply2(vds: VariantDataset): Array[Int] = {
    vds.rdd.map { case (v, (_, gs)) => (v.hashCode(), gs.map(_.isCalledNonRef)) }
      .aggregate(Array.fill[Int](vds.nSamples)(Int.MaxValue))({ case (a, (hash, calls)) =>
        calls.zip(a).map { case (b, i) => if (b) math.min(hash, i) else i }.toArray },
        (l, r) => (l, r).zipped.map(math.min))
  }

  def approxJacaardDist(mat: DenseMatrix[Int]): DenseMatrix[Double] = {
    val k = mat.rows
    val n = mat.cols
    val dist = DenseMatrix.zeros[Double](n,n)
    val inc = 1.0/k
    for {
      i <- 0 until k
      matches = Range(0,n).groupBy(mat(i,_))
      hash <- matches.keys
      a <- matches(hash)
      b <- matches(hash)
      if a != b
    } dist(a,b) += inc
    dist
  }

  def hashChain(k: Int, seed: Int): DenseVector[Int] = {
    val hashes = DenseVector.zeros[Int](k)
    hashes(0) = seed
    var i = 1
    while (i < k) {
      hashes(i) = MH3.finalizeHash(MH3.mixLast(MH3.arraySeed, hashes(i-1)), 0)
      i += 1
    }
    hashes
  }
}
