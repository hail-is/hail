package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils.MultiArray2
import org.testng.annotations.Test
import org.scalacheck._
import org.scalacheck.util.Buildable._
import org.scalacheck.Prop._
import org.scalacheck.util.Buildable
import org.scalacheck.Arbitrary._
import scala.language.implicitConversions

class MultiArray2Suite extends SparkSuite{
  @Test def test() = {

    val maxDimensionSize = 10

    def genOption[T](g:Gen[T]): Gen[Option[T]] = Gen.frequency((1, Gen.const(None)),(4, g.map(Some(_))))
    def genInt = Gen.choose(Int.MinValue,Int.MaxValue)
    def genString = Gen.alphaStr
    def genBoolean = Gen.oneOf(true,false)
    def genArray[T](n:Int,g:Gen[T])(implicit bT: Buildable[T,Array[T]]): Gen[Array[T]] = Gen.containerOfN[Array,T](n,g)

    def genMultiArray2[T](g:Gen[T])(implicit bT: Buildable[T,Array[T]]) = {
      for (n1 <- Gen.choose(0,maxDimensionSize);
           n2 <- Gen.choose(0,maxDimensionSize);
           a <- genArray[T](n1*n2,g)
      )
       yield new MultiArray2(n1,n2,a)
    }

    def genPairMultiArray2[T,S](g1:Gen[T],g2:Gen[S])(implicit bT: Buildable[T,Array[T]],bS:Buildable[S,Array[S]]) = {
      for (n1 <- Gen.choose(0,maxDimensionSize);
           n2 <- Gen.choose(0,maxDimensionSize);
           a1 <- genArray[T](n1*n2,g1);
           a2 <- genArray[S](n1*n2,g2)
      )
        yield (new MultiArray2(n1,n2,a1),new MultiArray2(n1,n2,a2))
    }

    def genNonPairMultiArray2[T,S](g1:Gen[T],g2:Gen[S])(implicit bT: Buildable[T,Array[T]],bS:Buildable[S,Array[S]]) = {
      for (n1 <- Gen.choose(0,maxDimensionSize);
           n2 <- Gen.choose(0,maxDimensionSize);
           n3 <- Gen.choose(0,maxDimensionSize);
           n4 <- Gen.choose(0,maxDimensionSize);
           a1 <- genArray[T](n1*n2,g1);
           a2 <- genArray[S](n3*n4,g2)
      )
        yield (new MultiArray2(n1,n2,a1),new MultiArray2(n3,n4,a2))
    }


    class MultiArray2Properties[T,S](val g1:Gen[T],val g2:Gen[S])
                                    (implicit bT:Buildable[T,Array[T]],bS:Buildable[S,Array[S]]) extends Properties("MultiArray2 Properties"){
      property("sizeNotNegative") = forAll(genInt,genInt){case (n1,n2) =>
          if (n1 < 0 || n2 < 0) throws(classOf[IllegalArgumentException])(new MultiArray2[Int](n1,n2,Array(0,0,0,0)))
          else true
      }
      property("sizeEqualsN1N2") = forAll(genMultiArray2(g1)) {ma => ma.n1*ma.n2 == ma.indices.size}
      property("indicesValid") = forAll(genMultiArray2(g1)) {ma => ma.indices.count{case(i,j) => i >= 0 && i < ma.n1 && j >= 0 && j < ma.n2} == ma.n1 * ma.n2}
      property("rowIndicesValid") = forAll(genMultiArray2(g1)) {ma => ma.rowIndices.count{case(i) => i >= 0 && i < ma.n1} == ma.n1}
      property("columnIndicesValid") = forAll(genMultiArray2(g1)) {ma => ma.columnIndices.count{case(j) => j >= 0 && j < ma.n2} == ma.n2}

      property("applyIndex") = forAll(genMultiArray2(g1)) { ma =>
        val n1 = Gen.choose(Int.MinValue,Int.MaxValue).sample.get
        val n2 = Gen.choose(Int.MinValue,Int.MaxValue).sample.get
        if (n1 < 0 || n1 > ma.n1 || n2 < 0 || n2 > ma.n2) throws(classOf[IllegalArgumentException])(ma(n1,n2))
        else true
      }

      property("applyRowIndex") = forAll(genMultiArray2(g1)) { ma =>
        val n1 = Gen.choose(Int.MinValue,Int.MaxValue).sample.get
        val n2 = Gen.choose(Int.MinValue,Int.MaxValue).sample.get
        if (n1 < 0 || n1 > ma.n1) throws(classOf[IllegalArgumentException])(ma.row(n1))
        else if (n2 < 0 || n2 > ma.row(n1).length) throws(classOf[ArrayIndexOutOfBoundsException])(ma.row(n1)(n2))
        else true
      }

      property("applyColumnIndex") = forAll(genMultiArray2(g1)) { ma =>
        val n1 = Gen.choose(Int.MinValue,Int.MaxValue).sample.get
        val n2 = Gen.choose(Int.MinValue,Int.MaxValue).sample.get
        if (n2 < 0 || n2 > ma.n2) throws(classOf[IllegalArgumentException])(ma.column(n2))
        else if (n1 < 0 || n1 > ma.column(n2).length) throws(classOf[ArrayIndexOutOfBoundsException])(ma.column(n2)(n1))
        else true
      }

      property("rowApplyIdentity") = forAll(genMultiArray2(g1)) {ma => ma.indices.count{case(i,j) => ma(i,j) == ma.row(i)(j)} == ma.n1*ma.n2}
      property("columnApplyIdentity") = forAll(genMultiArray2(g1)) {ma => ma.indices.count{case(i,j) => ma(i,j) == ma.column(j)(i)} == ma.n1*ma.n2}

      property("updateValues") = forAll(genMultiArray2(g1)) {ma => ma.indices.count{case(i,j) =>
        val randomValue = g1.sample.get
        ma(i,j) = randomValue
        ma(i,j) == randomValue} == ma.n1*ma.n2
      }

      property("zipSize") = forAll(genNonPairMultiArray2(g1,g2)) {case (ma1,ma2) =>
        if (ma1.n1 != ma2.n1 || ma1.n2 != ma2.n2) throws(classOf[IllegalArgumentException])(ma1.zip(ma2))
        else true
      }

      property("zipComposition") = forAll(genPairMultiArray2(g1,g2)) {case (ma1,ma2) =>
        val ma1Zip = ma1.zip(ma2)
        ma1Zip.indices.count{case(i,j) => ma1Zip(i,j) == (ma1(i,j),ma2(i,j))} == ma1Zip.n1 * ma1Zip.n2
      }

      property("zipCommutative") = forAll(genPairMultiArray2(g1,g2)) {case (ma1,ma2) =>
        val ma1Zip = ma1.zip(ma2)
        val ma2Zip = ma2.zip(ma1)
        ma1Zip.indices.count{case(i,j) =>
          val result2 = ma2Zip(i,j)
          ma1Zip(i,j) == (result2._2,result2._1)} == ma1Zip.n1 * ma1Zip.n2
      }
    }

    // Test MultiArray2 with combinations of various type generators
    val generatorMap = Map("Int" -> genInt,"String" -> genString, "Boolean" -> genBoolean,
      "Option[Int]" -> genOption(genInt), "Option[String]" -> genOption(genString),"Option[Boolean]" -> genOption(genBoolean))

    for ((t1,g1) <- generatorMap; (t2,g2) <- generatorMap){
      println(s"Using an $t1/$t2 Generator:")
      val test = new MultiArray2Properties(g1,g2)
      test.check
    }
  }
}
