package org.broadinstitute.hail.methods

import org.broadinstitute.hail.{ScalaCheckSuite, SparkSuite}
import org.broadinstitute.hail.utils.MultiArray2
import org.testng.annotations.Test
import org.scalacheck._
import org.scalacheck.util.Buildable._
import org.scalacheck.Prop.{throws,forAll,BooleanOperators}
import org.scalacheck.util.Buildable
import org.scalacheck.Arbitrary._
import scala.language.implicitConversions

object MultiArray2Suite {

  object Spec extends Properties("MultiArray2") {
/*    property("sizeNotNegative") = forAll { (n1: Int, n2: Int) => (n1 < 0 || n2 < 0) ==>
      throws(new MultiArray2[Int](n1,n2,Array(0,0,0,0))},classOf[IllegalArgumentException])
    }*/

    property("sizeEqualsN1N2") = forAll { ma: MultiArray2[Int] => ma.n1 * ma.n2 == ma.indices.size }
    property("indicesValid") = forAll { ma: MultiArray2[Int] => ma.indices.count { case (i, j) => i >= 0 && i < ma.n1 && j >= 0 && j < ma.n2 } == ma.n1 * ma.n2 }
    property("rowIndicesValid") = forAll { ma: MultiArray2[Int] => ma.rowIndices.count { case (i) => i >= 0 && i < ma.n1 } == ma.n1 }
    property("columnIndicesValid") = forAll { ma: MultiArray2[Int] => ma.columnIndices.count { case (j) => j >= 0 && j < ma.n2 } == ma.n2 }

    property("applyIndex") = forAll { ma: MultiArray2[Int] =>
      val n1 = Gen.choose(Int.MinValue, Int.MaxValue).sample.get
      val n2 = Gen.choose(Int.MinValue, Int.MaxValue).sample.get
      if (n1 < 0 || n1 > ma.n1 || n2 < 0 || n2 > ma.n2) throws(classOf[IllegalArgumentException])(ma(n1, n2))
      else true
    }

    property("applyRowIndex") = forAll { ma: MultiArray2[Int] =>
      val n1 = Gen.choose(Int.MinValue, Int.MaxValue).sample.get
      val n2 = Gen.choose(Int.MinValue, Int.MaxValue).sample.get
      if (n1 < 0 || n1 > ma.n1) throws(classOf[IllegalArgumentException])(ma.row(n1))
      else if (n2 < 0 || n2 > ma.row(n1).length) throws(classOf[ArrayIndexOutOfBoundsException])(ma.row(n1)(n2))
      else true
    }

    property("applyColumnIndex") = forAll { ma: MultiArray2[Int] =>
      val n1 = Gen.choose(Int.MinValue, Int.MaxValue).sample.get
      val n2 = Gen.choose(Int.MinValue, Int.MaxValue).sample.get
      if (n2 < 0 || n2 > ma.n2) throws(classOf[IllegalArgumentException])(ma.column(n2))
      else if (n1 < 0 || n1 > ma.column(n2).length) throws(classOf[ArrayIndexOutOfBoundsException])(ma.column(n2)(n1))
      else true
    }

    property("rowApplyIdentity") = forAll { ma: MultiArray2[Int] => ma.indices.count { case (i, j) => ma(i, j) == ma.row(i)(j) } == ma.n1 * ma.n2 }
    property("columnApplyIdentity") = forAll { ma: MultiArray2[Int] => ma.indices.count { case (i, j) => ma(i, j) == ma.column(j)(i) } == ma.n1 * ma.n2 }

    /* property("updateValues") = forAll {ma:MultiArray2 => ma.indices.count{case(i,j) =>
      val randomValue = g1.sample.get
      ma(i,j) = randomValue
      ma(i,j) == randomValue} == ma.n1*ma.n2
    }

    property("zipSize") = forAll {case (ma1:MultiArray2,ma2:MultiArray2) =>
      if (ma1.n1 != ma2.n1 || ma1.n2 != ma2.n2) throws(classOf[IllegalArgumentException])(ma1.zip(ma2))
      else true
    }

    property("zipComposition") = forAll {case (ma1:MultiArray2,ma2:MultiArray2) =>
      val ma1Zip = ma1.zip(ma2)
      ma1Zip.indices.count{case(i,j) => ma1Zip(i,j) == (ma1(i,j),ma2(i,j))} == ma1Zip.n1 * ma1Zip.n2
    }

    property("zipCommutative") = forAll {case (ma1:MultiArray2,ma2:MultiArray2) =>
      val ma1Zip = ma1.zip(ma2)
      val ma2Zip = ma2.zip(ma1)
      ma1Zip.indices.count{case(i,j) =>
        val result2 = ma2Zip(i,j)
        ma1Zip(i,j) == (result2._2,result2._1)} == ma1Zip.n1 * ma1Zip.n2
    }
  }*/
  }
}

class MultiArray2Suite extends SparkSuite with ScalaCheckSuite {
  import MultiArray2Suite._

  @Test def test() = {

    def genMultiArray2[T](g: Gen[T])(implicit bT: Buildable[T, Array[T]]) = {
      def genArray[T](n: Int, g: Gen[T])(implicit bT: Buildable[T, Array[T]]): Gen[Array[T]] = Gen.containerOfN[Array, T](n, g)
      val maxDimensionSize = 10
      for (n1 <- Gen.choose(0, maxDimensionSize);
           n2 <- Gen.choose(0, maxDimensionSize);
           a <- genArray[T](n1 * n2, g)
      )
        yield new MultiArray2(n1, n2, a)
    }

    implicit def arbMultiArray2[T](implicit a: Arbitrary[T], bT:Buildable[T,Array[T]]): Arbitrary[MultiArray2[T]] = Arbitrary(genMultiArray2(a.arbitrary))

    check(Spec)
  }
}
