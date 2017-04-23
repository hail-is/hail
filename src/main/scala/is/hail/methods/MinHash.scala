package is.hail.methods

import breeze.linalg._
import is.hail.variant.{Genotype, Variant, VariantDataset}
import is.hail.annotations.Annotation
import scala.collection.mutable.HashSet
import scala.util.hashing.{MurmurHash3 => MH3}
import scala.util.Random
import scala.collection.immutable.SortedMap

object MinHash {
  // returns a k by vds.nSamples matrix
  def apply(vds: VariantDataset, k: Int): DenseMatrix[Int] = {
    val n = vds.nSamples
    def seqOp(l: DenseMatrix[Int], r: (Variant, (Annotation, Iterable[Genotype]))): DenseMatrix[Int] = r match {
      case (_, (_, gs)) =>
        val gsItr = gs.hardCallGenotypeIterator
        val hashes = DenseVector.fill[Int](k)(Random.nextInt())
        var i = 0
        while(i < n) {
          //TODO: Write a min UFunc that modifies the left arg in place
          if (gsItr.next() > 0) l(::,i) := min(l(::,i), hashes)
          i += 1
        }
        l
    }
    val zeroValue = DenseMatrix.fill[Int](k, n)(Int.MaxValue)
    vds.rdd.aggregate(zeroValue)(seqOp, min(_,_))
  }

  //mat is a k-by-n matrix, which will be split into r-by-n submatrices for Min-LSH
  def findSimilarPairs(mat: DenseMatrix[Int], blockSize: Int): Set[(Int, Int)] =
    findSimilarPairs(mat, blockSize, mat.rows / blockSize)

  // specifying l might be helpful for trying different parameters without recomputing minHash
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
