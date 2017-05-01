package is.hail.methods

import breeze.linalg.{DenseVector, DenseMatrix}
import is.hail.SparkSuite
import org.testng.annotations.Test
import is.hail.stats
import scala.util.Random
import scala.util.hashing.MurmurHash3

class MinHashSuite extends SparkSuite {
  @Test def minHashTest() {
    val k = 4
    val genotypes =
      new DenseMatrix[Int](8, 3, Array[Int](
        1, 0, 0, 2, 1, -1, 2, 0,
        0, 1, 0, 1, 0,  2, 1, 0,
        0, 0, 1, 0, 1,  1, 1, 0)
      )

    val vds = stats.vdsFromMatrix(hc)(genotypes)

    val m = MinHash.fastMinHash(vds, k)
    for (i <- 0 until k) assert(m(i, 3) == math.min(m(i, 0), m(i, 1)))
    for (i <- 0 until k) assert(m(i, 4) == math.min(m(i, 0), m(i, 2)))
    for (i <- 0 until k) assert(m(i, 5) == math.min(m(i, 1), m(i, 2)))
    for (i <- 0 until k) assert(m(i, 6) == math.min(m(i, 0), math.min(m(i, 1), m(i, 2))))
    for (i <- 0 until k) assert(m(i, 7) == Int.MaxValue)
  }

  @Test def trueJacaardFromVDSTest() {
    val genotypes = new DenseMatrix[Int](8, 3,
      Array[Int](1, 0, 0, 2, 1, -1, 2, 0,
                 0, 1, 0, 1, 0,  2, 1, 0,
                 0, 0, 1, 0, 1,  1, 1, 0))

    val vds = stats.vdsFromMatrix(hc)(genotypes)

    val jacaardDist = DenseMatrix(
      (  1.0,   0.0,   0.0,   0.5,   0.5,   0.0, 1.0/3, 0.0),
      (  0.0,   1.0,   0.0,   0.5,   0.0,   0.5, 1.0/3, 0.0),
      (  0.0,   0.0,   1.0,   0.0,   0.5,   0.5, 1.0/3, 0.0),
      (  0.5,   0.5,   0.0,   1.0, 1.0/3, 1.0/3, 2.0/3, 0.0),
      (  0.5,   0.0,   0.5, 1.0/3,   1.0, 1.0/3, 2.0/3, 0.0),
      (  0.0,   0.5,   0.5, 1.0/3, 1.0/3,   1.0, 2.0/3, 0.0),
      (1.0/3, 1.0/3, 1.0/3, 2.0/3, 2.0/3, 2.0/3,   1.0, 0.0),
      (  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0)
    )

    assert(MinHash.trueJacaardFromVDS(vds) == jacaardDist)
  }

  @Test def jacaardDistTest() {
    val minHash = DenseMatrix(
      ( 1237213332,   259873309, -1445737267,   259873309, -1445737267, -1445737267, -1445737267),
      (  687582406, -1252237373, -1251331505, -1252237373, -1251331505, -1252237373, -1252237373),
      ( -665338010,  1340670822,  -486674642,  -665338010,  -665338010,  -486674642,  -665338010),
      (-1175354349,  1291508146,  1728036926, -1175354349, -1175354349,  1291508146, -1175354349)
    )
    val jDist = DenseMatrix(
      ( 1.0, 0.0,  0.0,  0.5,  0.5,  0.0,  0.5),
      ( 0.0, 1.0,  0.0,  0.5,  0.0,  0.5,  0.25),
      ( 0.0, 0.0,  1.0,  0.0,  0.5,  0.5,  0.25),
      ( 0.5, 0.5,  0.0,  1.0,  0.5,  0.25, 0.75),
      ( 0.5, 0.0,  0.5,  0.5,  1.0,  0.25, 0.75),
      ( 0.0, 0.5,  0.5,  0.25, 0.25, 1.0,  0.5),
      ( 0.5, 0.25, 0.25, 0.75, 0.75, 0.5,  1.0)
    )
    assert(MinHash.approxJacaardDistance(minHash) == jDist)
  }

  @Test def simPairsTest() {
    val minHash = DenseMatrix(
      ( 1237213332,   259873309, -1445737267,   259873309, -1445737267, -1445737267, -1445737267),
      (  687582406, -1252237373, -1251331505, -1252237373, -1251331505, -1252237373, -1252237373),
      ( -665338010,  1340670822,  -486674642,  -665338010,  -665338010,  -486674642,  -665338010),
      (-1175354349,  1291508146,  1728036926, -1175354349, -1175354349,  1291508146, -1175354349)
    )
    val simPairs = Set((0,3), (0,4), (0,6), (1,3), (2,4), (3,4), (3,6), (4,6), (5,6))
    assert(MinHash.findSimilarPairs(minHash, 2) == simPairs)
  }

  @Test def trueJacaardTest() {
    val genotypes = new DenseMatrix[Int](8, 3,
      Array[Int](1, 0, 0, 2, 1, -1, 2, 0,
                 0, 1, 0, 1, 0,  2, 1, 0,
                 0, 0, 1, 0, 1,  1, 1, 0))
    val jacaardDist = DenseMatrix(
      (  1.0,   0.0,   0.0,   0.5,   0.5,   0.0, 1.0/3, 0.0),
      (  0.0,   1.0,   0.0,   0.5,   0.0,   0.5, 1.0/3, 0.0),
      (  0.0,   0.0,   1.0,   0.0,   0.5,   0.5, 1.0/3, 0.0),
      (  0.5,   0.5,   0.0,   1.0, 1.0/3, 1.0/3, 2.0/3, 0.0),
      (  0.5,   0.0,   0.5, 1.0/3,   1.0, 1.0/3, 2.0/3, 0.0),
      (  0.0,   0.5,   0.5, 1.0/3, 1.0/3,   1.0, 2.0/3, 0.0),
      (1.0/3, 1.0/3, 1.0/3, 2.0/3, 2.0/3, 2.0/3,   1.0, 0.0),
      (  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 0.0)
    )

    assert(MinHash.trueJacaardDistance(genotypes.map(v => if (v > 0) 1 else 0)) == jacaardDist)
  }
}