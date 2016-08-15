package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils._

import scala.io.Source

class WardSuite extends SparkSuite {

  @Test def test() {

    val W = new Ward()
    
    // Matrices for distMat
    val M1 = Matrices.dense(3,3,Array(1.0,1.0,0.0,-1.0,0.0,-1.0,5.5,5.5,4.5)) // Yields D1
   
    // Distance constructs for NN, RNN, and apply
    val D1 = Array(Array(1.0,2.0),Array(3.0)) // Trivial
    val D2 = Array(Array(1.0,18.0,32.0),Array(8.0,18.0),Array(1.0)) // Two pairs of close points   
    
    // Hierarchies for cut
    val H1 = Set((0,1,1.0),(0,2,4.0))
    val H2 = Set((0,3,2.0),(0,5,2.5),(1,2,1.0),(0,1,3.0),(3,4,0.5))
  
    // Test distMat
    assert(W.distMat(M1).deep == D1.deep)

    // Test getDist
    assert(W.getDist(D1,0,1) == 1.0)
    assert(W.getDist(D1,0,2) == 2.0)
    assert(W.getDist(D1,1,2) == 3.0)
    assert(W.getDist(D1,0,1) == W.getDist(D1,1,0))
    assert(W.getDist(D2,3,2) == W.getDist(D2,2,3))

    // Test LW
    assert(D_==(W.LW(D1,0,1,2,Vector(1,1,1)),3.0))
    assert(D_==(W.LW(D1,0,1,2,Vector(8,2,3)),2.615384615384615))
    assert(D_==(W.LW(D1,1,0,2,Vector(8,2,3)),W.LW(D1,0,1,2,Vector(8,2,3))))

    // Test LWupdate
    val D1test = W.distMat(M1)
    val D1update = W.LWupdate(D1test,0,1,Vector(1,1,1),Set(0,2))
    assert(D_==(D1update(0)(1), 3.0))
    assert(D_==(D1update(0)(0), D1(0)(0)))
    assert(D_==(D1update(1)(0), D1(1)(0)))

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
 
    // Test makeDendro
    val C1_1 = W.makeClusters(H1,3)
    val C_3 = W.makeClusters(Set((0,3,2.0),(1,5,1.0),(0,1,3.0),(2,4,0.5)),7)  
    assert(C1_1.size == 1)
    assert(C1_1 contains 0)
    assert(C1_1(0).size == 3)
    assert(C1_1(0) == Set(0,1,2))
    assert(C_3.size == 3)
    assert(C_3(6) == Set(6))
    assert(C_3(0) == Set(0,1,3,5))
    assert(C_3(2) == Set(2,4))

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
    val points : Array[Double] = readFile("src/test/resources/wardSuitePoints.txt", sc.hadoopConfiguration) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => line.length > 1)
        .map(line => {
          val split = line.split(" ")
          split map (x => x.toDouble)
          }).flatten.toArray
      }
    val answers_2 : Set[Set[Int]] = readFile("src/test/resources/wardSuiteClusters2.txt", sc.hadoopConfiguration) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => line.length > 1)
        .map(line => {
          val split = line.split(" ")
          (split map (x => x.toInt)).toSet
          }).toSet
      }
    val pointsMat = Matrices.dense(10,100,points).transpose
    val Dlarge = W.distMat(pointsMat)
    
    assert(D_==(W.pointDist(pointsMat,0,1),12799.25))
    assert(D_==(W.pointDist(pointsMat,0,1),W.getDist(Dlarge,0,1)))
    
    val clustlarge_2 = W(Dlarge,2).toSet
    
    assert(clustlarge_2 == answers_2)
  }
}
