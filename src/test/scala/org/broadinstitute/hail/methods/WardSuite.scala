package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils._

class WardSuite extends SparkSuite {

  @Test def test() {

    val W = new Ward()
    
    // Matrices for distMat
    val M1 = Matrices.dense(3,3,Array(1.0,1.0,0.0,-1.0,0.0,-1.0,5.5,5.5,4.5)) // Yields D1
   
    // Distance constructs for NN, RNN, and apply
    val D1 = Vector(Vector(1.0,2.0),Vector(3.0)) // Trivial
    val D2 = Vector(Vector(1.0,18.0,32.0),Vector(8.0,18.0),Vector(1.0)) // Two pairs of close points   
    
    // Hierarchies for cut
    val H1 = Set((0,1,1.0),(0,2,4.0))
    val H2 = Set((0,3,2.0),(0,5,2.5),(1,2,1.0),(0,1,3.0),(3,4,0.5))
  
    // Test distMat
    assert(W.distMat(M1) == D1)

    // Test NN and RNN
    assert(W.NN(D1,0,Set(0,1,2)) == (1.0,1))
    assert(W.NN(D1,1,Set(0,1,2)) == (1.0,0))
    assert(W.NN(D1,2,Set(0,1,2)) == (2.0,0))
    assert(W.NN(D1,0,Set(0,1)) == (1.0,1))
    assert(W.NN(D1,1,Set(0,1)) == (1.0,0))
    assert(W.NN(D1,2,Set(1,2)) == (3.0,1))
    assert(W.RNN(D1,0,Set(0,1,2)) == (0,1,1.0))
    assert(W.RNN(D1,1,Set(0,1,2)) == (0,1,1.0))
    assert(W.RNN(D1,2,Set(0,1,2)) == (0,1,1.0))

    assert(W.NN(D2,0,Set(0,1,2,3)) == (1.0,1))
    assert(W.NN(D2,0,Set(0,2,3)) == (18.0,2))
    assert(W.NN(D2,2,Set(0,1,2,3)) == (1.0,3))
    assert(W.RNN(D2,0,Set(0,1,2,3)) == (0,1,1.0))
    assert(W.RNN(D2,3,Set(0,1,2,3)) == (2,3,1.0))
    assert(W.RNN(D2,0,Set(0,2,3)) == (2,3,1.0))

    // Test maxJoin
    assert(W.maxJoin(H1) == (0,2,4.0))
    assert(W.maxJoin(H2) == (0,1,3.0))
    
    // Test makeClusters
    val C1_1 = W.makeClusters(H1,3)
    assert(C1_1.size == 1)
    assert(C1_1 contains 0)
    assert(C1_1(0).size == 3)
    assert(C1_1(0) contains 0)
    assert(C1_1(0) contains 1)
    assert(C1_1(0) contains 2)

    // Test cut
    val C1_2 = W.cut(H1,2,3)
    val C2_2 = W.cut(H2,2,6)
    val C2_3 = W.cut(H2,3,6)
    val C2_6 = W.cut(H2,6,6)
    assert(C1_2.size == 2)
    assert(C1_2 contains 0)
    assert(C1_2 contains 2)
    assert(C1_2(0).size == 2)
    assert(C1_2(2).size == 1)
    assert(C2_2.size == 2)
    assert(C2_2 contains 0)
    assert(C2_2 contains 1)
    assert(C2_2(0) contains 5)
    assert(C2_2(0) contains 4)
    assert(C2_2(0) contains 3)
    assert(C2_2(1) contains 2)
    assert(C2_3.size == 3)
    assert(C2_3 contains 0)
    assert(C2_3 contains 1)
    assert(C2_3 contains 5)
    assert(C2_3(0).size == 3)
    assert(C2_3(1).size == 2)
    assert(C2_3(5).size == 1)
    assert(C2_6.size == 6)

    // Test apply()
    val clust1_1 = W(D1,1)
    val clust1_2 = W(D1,2)
    val clust1_3 = W(D1,3)
    val clust2_2 = W(D2,2)
    assert(clust1_2.size == 2)
    assert(clust1_1.size == 1)
    assert(clust1_3.size == 3)
    assert(clust1_2 contains Set(0,1))
    assert(clust1_2 contains Set(2))
    assert(clust1_1 contains Set(0,1,2))
    assert(clust2_2 contains Set(0,1))
    assert(clust2_2 contains Set(2,3))

    // Bigger test cases
    /*val fileMap = readFile("src/test/resources/points1.tsv", sc.hadoopConfiguration) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.startsWith("Sample"))
        .map(line => {
          val split = line.split("\t")
          val f3 = split(2) match {
            case "NA" => None
            case x => Some(x.toInt)
          }
          (split(0), (Some(split(1)), f3))
        })
        .toMap
    }*/

  }

}
