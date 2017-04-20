package is.hail.methods

import breeze.linalg.{DenseVector, DenseMatrix}
import is.hail.SparkSuite
import org.testng.annotations.Test
import is.hail.stats
import scala.util.hashing.MurmurHash3

class MinHashSuite extends SparkSuite {
  @Test def test() = {
    val k = 6
    val genotypes = new DenseMatrix[Int](4, 3,
      Array[Int](0, 1, 0, -1,
                 0, 0, 2,  2,
                 2, 1, 1, -1))

    val vds = stats.vdsFromMatrix(hc)(genotypes)

    for ((v,i) <- vds.variants.zipWithIndex) {
      println(s"V$i: ${MinHash.hashChain(k,v.hashCode())}")
    }
    val minHash = MinHash.kMinHash(vds,k)
    println("Min hashes:")
    println(minHash)
    println("Jacaard dist:")
    println(MinHash.approxJacaardDist(minHash))
    print("M-LSH pairs: ")
    for ((a,b) <- MinHash.findSimilarPairs(minHash,2)) print(s"($a,$b), ")
    println()

    assert(true)
  }
}