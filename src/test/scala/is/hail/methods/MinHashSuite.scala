package is.hail.methods

import breeze.linalg.DenseMatrix
import is.hail.SparkSuite
import org.testng.annotations.Test
import is.hail.stats
import scala.util.hashing.MurmurHash3

class MinHashSuite extends SparkSuite {
  @Test def test() = {
    val genotypes = new DenseMatrix[Int](4, 3,
      Array[Int](0, 1, 0, -1,
                 0, 0, 2,  2,
                 2, 1, 1, -1))

    val vds = stats.vdsFromMatrix(hc)(genotypes)

    for ((v,i) <- vds.variants.zipWithIndex) {
      println(s"V$i: ${MinHash.hashChain(3,v.hashCode())}")
    }
    val minHash = MinHash.kMinHash(vds,3)
    println("Min hashes:")
    println(minHash)
    println("Jacaard dist:")
    println(MinHash.approxJacaardDist(minHash))

    assert(true)
  }
}