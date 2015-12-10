package org.broadinstitute.hail.methods

import org.broadinstitute.hail.{ScalaCheckSuite, SparkSuite}
import org.broadinstitute.hail.utils.MultiArray2
import org.testng.annotations.Test
import org.scalacheck._
import org.scalacheck.util.Buildable._
import org.scalacheck.Prop.{throws, forAll, BooleanOperators, classify}
import org.scalacheck.Arbitrary._
import scala.language.implicitConversions



object MultiArray2Suite {
  import MultiArray2._

  object Spec extends Properties("MultiArray2") {

    property("sizeNotNegative") = forAll(Gen.negNum[Int],Gen.negNum[Int]) { (n1:Int, n2:Int) =>
      (n1 < 0 || n2 < 0) ==> throws(classOf[IllegalArgumentException])(new MultiArray2[Int](n1, n2, Array(0, 0, 0, 0)))
    }

    property("sizeEqualsN1N2") = forAll { ma:MultiArray2[Int] => ma.n1 * ma.n2 == ma.indices.size}

    property("indicesValid") = forAll { ma:MultiArray2[Int] => ma.indices.forall {case (i,j) => i >= 0 && i < ma.n1 && j >= 0 && j < ma.n2}}

    property("rowIndices") = forAll { ma:MultiArray2[Int] => ma.rowIndices.forall {case (i) => i >= 0 && i < ma.n1}}

    property("columnIndices") = forAll { ma:MultiArray2[Int] => ma.columnIndices.forall {case (j) => j >= 0 && j < ma.n2}}

    property("applyIdentity") = forAll { ma:MultiArray2[Int] => ma.indices.forall {case (i,j) => ma(i,j) == ma.row(i)(j) && ma(i,j) == ma.column(j)(i)}}

    property("updateIdentity") = forAll {(ma:MultiArray2[Int]) => ma.indices.forall{case (i,j) => ma(i,j) = 100; ma(i,j) == 100}}

    property("apply") = forAll { (ma:MultiArray2[Int], i:Int, j:Int) =>
      classify(i >= 0 && i < ma.n1 && j >= 0 && j < ma.n2,"IndexInBounds","IndexOutOfBounds") {
        if (i < 0 || i >= ma.n1 || j < 0 || j >= ma.n2)
          throws(classOf[ArrayIndexOutOfBoundsException])(ma(i, j))
        else
          ma(i, j) == ma.row(i)(j) && ma(i, j) == ma.column(j)(i)
      }
    }

    property("zip") = forAll { (ma1:MultiArray2[Int], ma2:MultiArray2[String]) =>
      classify(ma1.n1 == ma2.n1 && ma1.n2 == ma2.n2,"SameShapes","DifferentShapes") {
        if (ma1.n1 != ma2.n1 || ma1.n2 != ma2.n2)
          throws(classOf[IllegalArgumentException])(ma1.zip(ma2))
        else {
          val ma1Zip = ma1.zip(ma2)
          val ma2Zip = ma2.zip(ma1)
          ma1Zip.indices.forall { case (i, j) => ma1Zip(i, j) ==(ma1(i, j), ma2(i, j)) && ma2Zip(i, j) ==(ma2(i, j), ma1(i, j)) }
        }
      }
    }

    property("zipIdentity") = forAll(Gen.choose[Int](0,10),Gen.choose[Int](0,10)) { (n1:Int, n2:Int) =>
      forAll(genMultiArray2Sized[Int](n1,n2),genMultiArray2Sized[String](n1,n2)) {
        (ma1:MultiArray2[Int], ma2:MultiArray2[String]) =>
          val ma1Zip = ma1.zip(ma2)
          val ma2Zip = ma2.zip(ma1)
          ma1Zip.indices.forall{case (i,j) => ma1Zip(i,j) == (ma1(i,j), ma2(i,j)) && ma2Zip(i,j) == (ma2(i,j),ma1(i,j))}
      }
    }
  }
}

class MultiArray2Suite extends SparkSuite with ScalaCheckSuite {

  import MultiArray2Suite._

  @Test def test() = {
    check(Spec)
  }
}
