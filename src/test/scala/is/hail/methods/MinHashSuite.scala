package is.hail.methods

import breeze.linalg.DenseMatrix
import is.hail.SparkSuite
import org.testng.annotations.Test
import is.hail.stats

class MinHashSuite extends SparkSuite {
  @Test def test() {
    println("Hello, world")
  }

  @Test def test2() = {
    val genotypes = new DenseMatrix[Int](4, 3,
      Array[Int](0, 1, 0, -1,
                 0, 0, 0,  2,
                 2, 1, 1, -1))

    val vds = stats.vdsFromMatrix(hc)(genotypes)

    println("Variant hashes: ", vds.rdd.map{ case (v, _) => v.hashCode() }.collect.mkString("[",", ","]"))
    println("Min hashes 1: ", MinHash(vds).mkString("[",", ","]"))
    println("Min hashes 2: ", MinHash.apply2(vds).mkString("[",", ","]"))

    assert(true)
  }
}