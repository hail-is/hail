package is.hail.methods

import breeze.linalg._
import is.hail.variant.{Genotype, Variant, VariantDataset}
import is.hail.annotations.Annotation
import scala.collection.mutable.HashSet
import scala.util.hashing.{MurmurHash3 => MH3}
import scala.util.Random
import scala.collection.immutable.SortedMap

object MinHash {
  // returns a nHashes by vds.nSamples matrix
  def apply(vds: VariantDataset, nHashes: Int): DenseMatrix[Int] = {
    val n = vds.nSamples
    def seqOp(l: DenseMatrix[Int], r: (Variant, (Annotation, Iterable[Genotype]))): DenseMatrix[Int] = r match {
      case (_, (_, gs)) =>
        val gsItr = gs.hardCallGenotypeIterator
        val hashes = DenseVector.fill[Int](nHashes)(Random.nextInt())
        var i = 0
        while(i < n) {
          if (gsItr.next() > 0) l(::,i) := min(l(::,i), hashes)
          i += 1
        }
        l
    }
    val zeroValue = DenseMatrix.fill[Int](nHashes, n)(Int.MaxValue)
    vds.rdd.aggregate(zeroValue)(seqOp, min(_,_))
  }

  def fastMinHash(vds: VariantDataset, nHashes: Int): DenseMatrix[Int] = {
    val n = vds.nSamples
    val matSize = n * nHashes
    def seqOp(l: Array[Int], r: (Variant, (Annotation, Iterable[Genotype]))): Array[Int] = r match {
      case (_, (_, gs)) =>
        val gsItr = gs.hardCallGenotypeIterator
        val hashes = Array.fill[Int](nHashes)(Random.nextInt())
        var i = 0
        while(i < n) {
          if (gsItr.next() > 0) {
            var j = 0
            while(j < nHashes) {
              l(i * nHashes + j) = scala.math.min(l(i * nHashes + j), hashes(j))
              j += 1
            }
          }
          i += 1
        }
        l
    }
    def combOp(l: Array[Int], r: Array[Int]): Array[Int] = {
      var i = 0
      while(i < matSize) {
        l(i) = scala.math.min(l(i), r(i))
        i += 1
      }
      l
    }
    val zeroValue = Array.fill[Int](matSize)(Int.MaxValue)
    val data = vds.rdd.aggregate(zeroValue)(seqOp, combOp)
    new DenseMatrix[Int](nHashes, n, data)
  }

  //mat is a nHashes-by-nSamples matrix, which will be split into blockSize-by-nSamples submatrices for Min-LSH
  def findSimilarPairs(mat: DenseMatrix[Int], blockSize: Int): Set[(Int, Int)] =
    findSimilarPairs(mat, blockSize, mat.rows / blockSize)

  // specifying numBlocks might be helpful for trying different parameters without recomputing minHash
  def findSimilarPairs(mat: DenseMatrix[Int], blockSize: Int, numBlocks: Int): Set[(Int, Int)] = {
    require(blockSize * numBlocks <= mat.rows)
    val n = mat.cols
    val simPairs = HashSet[(Int, Int)]()
    for {
      i <- 0 until numBlocks
      matches = Range(0, n).groupBy(j =>
        MH3.orderedHash( mat(i * blockSize until (i + 1) * blockSize, j).valuesIterator )
      )
      hash <- matches.keys
      a <- matches(hash)
      b <- matches(hash)
      if a < b
    } simPairs += ((a, b))
    simPairs.toSet
  }

  // takes k-by-n matrix, returns n-by-n symmetric similarity matrix
  def approxJacaardDistance(mat: DenseMatrix[Int]): DenseMatrix[Double] = {
    val k = mat.rows
    val n = mat.cols
    val dist = DenseMatrix.zeros[Double](n, n)
    val inc = 1.0 / k
    for {
      i <- 0 until k
      matches = Range(0,n).groupBy(mat(i, _))
      hash <- matches.keys
      a <- matches(hash)
      b <- matches(hash)
    } dist(a, b) += inc
    dist
  }

  // takes a binary n-by-m matrix, returns an n-by-n symmetric similarity matrix
  def trueJacaardDistance(mat: DenseMatrix[Int]): DenseMatrix[Double] = {
    val n = mat.rows
    val m = mat.cols

    val counts = sum(mat(*, ::))
    val intersections = DenseMatrix.zeros[Double](n,n)
    for {
      j <- 0 until m
      occurrences = Range(0, n).filter(mat(_, j) == 1)
      a <- occurrences
      b <- occurrences
    } intersections(a, b) += 1

    for (i <- 0 until n; j <- 0 until n; if intersections(i, j) != 0)
      intersections(i, j) /= counts(i) + counts(j) - intersections(i, j)

    intersections
  }

  def trueJacaardFromVDS(vds: VariantDataset): DenseMatrix[Double] = {
    val n = vds.nSamples
    val rdd = vds.rdd.map(t => t._2._2)
    val zeroValue = DenseMatrix.zeros[Int](n, n)

    def seqOp(counts: DenseMatrix[Int], gs: Iterable[Genotype]): DenseMatrix[Int] = {
      val gsItr = gs.hardCallGenotypeIterator
      var i = 0
      while (i < n) {
        if (gsItr.next() > 0) {
          val gsItr2 = gs.hardCallGenotypeIterator
          var j = 0
          while (j < n) {
            if (gsItr2.next() > 0) counts(i, j) += 1
            j += 1
          }
        }
        i += 1
      }
      counts
    }

    val distances = rdd.aggregate(zeroValue)(seqOp, _ + _).mapValues(_.toDouble)
    for (i <- 0 until n; j <- 0 until n if i != j && distances(i, j) != 0)
      distances(i, j) /= distances(i, i) + distances(j, j) - distances(i, j)
    for (i <- 0 until n if distances(i, i) != 0) distances(i, i) = 1
    distances
  }

  def hitProb(sim: Double, numBlocks: Int, blockSize: Int): Double =
    1 - math.pow(1 - math.pow(sim, blockSize), numBlocks)

  def falseNegRate(blockSize: Int, numBlocks: Int, simThresh: Double, simDist: SortedMap[Double, Double]): Double =
    simDist.from(simThresh).map{case (sim, rate) => rate * (1 - hitProb(sim, numBlocks, blockSize))}.sum

  def falsePosRate(blockSize: Int, numBlocks: Int, simThresh: Double, simDist: SortedMap[Double, Double]): Double =
    simDist.until(simThresh).map{case (sim, rate) => rate * hitProb(sim, numBlocks, blockSize)}.sum

  def optimizeNumBlocks(blockSize: Int, falseNegThresh: Double,
                        simThresh: Double, simDist: SortedMap[Double, Double]): Int = {

    // maintain falseNegRate(numBlocksBot, blockSize) > falseNegThresh,
    // falseNegRate(numBlocksTop, blockSize) <= falseNegThresh
    def binarySearch(numBlocksBot: Int, numBlocksTop: Int, blockSize: Int): Int = {
      if (numBlocksTop - numBlocksBot <= 1)
        numBlocksBot
      else {
        val mid = (numBlocksBot + numBlocksTop) / 2
        if (falseNegRate(blockSize, mid, simThresh, simDist) <= falseNegThresh)
          binarySearch(numBlocksBot, mid, blockSize)
        else
          binarySearch(mid, numBlocksTop, blockSize)
      }
    }

    val start = ((math.pow(simThresh, blockSize) + blockSize - 1) /
                 (blockSize * math.pow(simThresh, blockSize))).toInt
    var i = 1
    while (falseNegRate(blockSize, i * start, simThresh, simDist) > falseNegThresh) i *= 2
    binarySearch((i / 2) * start, i * start, blockSize)
  }
}
