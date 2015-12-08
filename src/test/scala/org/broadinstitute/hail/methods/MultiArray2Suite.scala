package org.broadinstitute.hail.methods

import org.broadinstitute.hail.{ScalaCheckSuite, SparkSuite}
import org.broadinstitute.hail.utils.MultiArray2
import org.testng.annotations.Test
import org.scalacheck._
import org.scalacheck.util.Buildable._
import org.scalacheck.Prop.{throws, forAll, BooleanOperators}
import org.scalacheck.Arbitrary._
import scala.language.implicitConversions

object MultiArray2Suite {

  object Spec extends Properties("MultiArray2") {

    property("sizeNotNegative") = forAll { (n1:Int, n2:Int) =>
      (n1 < 0 || n2 < 0) ==> throws(classOf[IllegalArgumentException])(new MultiArray2[Int](n1, n2, Array(0, 0, 0, 0)))
    }

    property("sizeEqualsN1N2") = forAll { ma:MultiArray2[Int] => ma.n1 * ma.n2 == ma.indices.size }
    property("indicesValid") = forAll { ma:MultiArray2[Int] => ma.indices.count { case (i, j) => i >= 0 && i < ma.n1 && j >= 0 && j < ma.n2 } == ma.n1 * ma.n2 }
    property("rowIndicesValid") = forAll { ma:MultiArray2[Int] => ma.rowIndices.count { case (i) => i >= 0 && i < ma.n1 } == ma.n1 }
    property("columnIndicesValid") = forAll { ma:MultiArray2[Int] => ma.columnIndices.count { case (j) => j >= 0 && j < ma.n2 } == ma.n2 }


    property("apply") = forAll { (ma:MultiArray2[Int], i:Int, j:Int) =>
      (i < 0 || i >= ma.n1 || j < 0 || j >= ma.n2) ==> throws(classOf[IllegalArgumentException])(ma(i, j))
    }
    property("applyRow") = forAll { (ma:MultiArray2[Int], i:Int) =>
      (i < 0 || i >= ma.n1) ==> throws(classOf[IllegalArgumentException])(ma.row(i))
    }
    property("applyRowIndex") = forAll { (ma:MultiArray2[Int], i:Int, j:Int) =>
      ((i >= 0 && i < ma.n1) && (j < 0 || j >= ma.n2)) ==> throws(classOf[ArrayIndexOutOfBoundsException])(ma.row(i)(j))
    }
    property("applyColumn") = forAll { (ma:MultiArray2[Int], j:Int) =>
      (j < 0 || j >= ma.n2) ==> throws(classOf[IllegalArgumentException])(ma.column(j))
    }
    property("applyColumnIndex") = forAll { (ma:MultiArray2[Int], i:Int, j:Int) =>
      ((j >= 0 && j < ma.n2) && (i < 0 || i >= ma.n1)) ==> throws(classOf[ArrayIndexOutOfBoundsException])(ma.column(j)(i))
    }

    property("rowApplyIdentity") = forAll { ma:MultiArray2[Int] => ma.indices.count { case (i, j) => ma(i, j) == ma.row(i)(j) } == ma.n1 * ma.n2 }
    property("columnApplyIdentity") = forAll { ma:MultiArray2[Int] => ma.indices.count { case (i, j) => ma(i, j) == ma.column(j)(i) } == ma.n1 * ma.n2 }

    property("updateValues") = forAll { (ma: MultiArray2[Int], i: Int, j: Int, randomValue: Int) =>
      (i >= 0 && i < ma.n1 && j >= 0 && j < ma.n2) ==> {
        ma(i, j) = randomValue
        ma(i, j) == randomValue
      }
    }

    property("zipSize") = forAll { (ma1:MultiArray2[Int], ma2:MultiArray2[String]) =>
      (ma1.n1 != ma2.n1 || ma1.n2 != ma2.n2) ==> throws(classOf[IllegalArgumentException])(ma1.zip(ma2))
    }

    property("zipComposition") = forAll { (ma1:MultiArray2[Int], ma2:MultiArray2[String]) =>
      (ma1.n1 == ma2.n1 && ma1.n2 == ma2.n2) ==> {
        val ma1Zip = ma1.zip(ma2)
        ma1Zip.indices.count { case (i, j) => ma1Zip(i, j) == (ma1(i, j), ma2(i, j)) } == ma1Zip.n1 * ma1Zip.n2
      }
    }

    property("zipComposition") = forAll { (ma1:MultiArray2[Int], ma2:MultiArray2[String]) =>
      (ma1.n1 == ma2.n1 && ma1.n2 == ma2.n2) ==> {
        val ma1Zip = ma1.zip(ma2)
        ma1Zip.indices.count { case (i, j) => ma1Zip(i, j) == (ma1(i, j), ma2(i, j)) } == ma1Zip.n1 * ma1Zip.n2
      }
    }

    property("zipCommutative") = forAll { (ma1:MultiArray2[Int], ma2:MultiArray2[String]) =>
      (ma1.n1 == ma2.n1 && ma1.n2 == ma2.n2) ==> {
        val ma1Zip = ma1.zip(ma2)
        val ma2Zip = ma2.zip(ma1)
        ma1Zip.indices.count { case (i, j) =>
          val result2 = ma2Zip(i, j)
          ma1Zip(i, j) ==(result2._2, result2._1)
        } == ma1Zip.n1 * ma1Zip.n2
      }
    }
  }
}

class MultiArray2Suite extends SparkSuite with ScalaCheckSuite {

  import MultiArray2Suite._

  @Test def test() = {
    Spec.check
    check(Spec)
  }
}
