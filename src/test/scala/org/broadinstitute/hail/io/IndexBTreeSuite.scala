package org.broadinstitute.hail.io

import org.broadinstitute.hail.{ScalaCheckSuite, SparkSuite}
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils.{hadoopOpen, hadoopCreate}
import org.scalacheck._
import org.scalacheck.util.Buildable._
import org.scalacheck.Prop.{throws, forAll, BooleanOperators, classify}
import org.scalacheck.Arbitrary._
import scala.language.implicitConversions

class IndexBTreeSuite extends SparkSuite with ScalaCheckSuite {
  val FILENAME = "/tmp/IndexBTreeTest.idx"
  val positiveLong = Gen.choose(1L, Long.MaxValue)
  val testGen = for (
    arr <- Gen.buildableOfN[Array[Long], Long](100, positiveLong).suchThat(arr => arr.length > 0);
    start <- Gen.choose(0L, arr.max);
    end <- Gen.choose(start, arr.max)) yield {
    (arr.map(l => (l, 0)).groupBy(_._1).keys.toArray.sorted, start, end)
  }

  object Spec extends Properties("IndexBTree"){

    property("writeQuery") = forAll(testGen) {
      query: (Array[Long], Long, Long) =>
        val arr = query._1
        val start = query._2
        val end = query._3

        IndexBTree.write(arr, hadoopCreate(FILENAME, sc.hadoopConfiguration))

        val result = arr.flatMap(l => if (start <= l && end >= l) Some(l) else None)
        val queryResult = IndexBTree.query(0, 50000, hadoopOpen(FILENAME, sc.hadoopConfiguration))
        result.sameElements(queryResult)
    }
  }

  @Test def test() {
    val hadoopConf = sc.hadoopConfiguration

//    val a1 = (0 until 500).map{ i => (i*10000).toLong}.toArray
//    val a2 = (0 until 500000).map{ i => (i*100).toLong}.toArray
//    val a3 = (0 until 5000000).map{ i => i.toLong}.toArray
//    val a1w = IndexBTree.write(a1, "src/test/resources/500Array.idx")
//    val a2w = IndexBTree.write(a2, "src/test/resources/500KArray.idx")
//    val a3w = IndexBTree.write(a3, "src/test/resources/5MArray.idx")

//    println
//    println("about to read 500k")

//    println("ret is: " + IndexBTree.query(200000, 205010, hadoopOpen("src/test/resources/500KArray.idx", hadoopConf)).mkString(","))
//    println(IndexBTree.queryArr(400050, 400500, a2w).mkString(","))
//    println(IndexBTree.queryArr(400050, 400095, a3w).mkString(","))


    // choose size, choose elements, sort,
    val positiveLong = Gen.choose(1L, Long.MaxValue)

//    val lengthGen = Gen.choose(1, 1e8.toInt)

    def printSample(query: Option[(Array[Long], Long, Long)]) {
      query match {
        case Some((arr: Array[Long], start: Long, end: Long)) =>
          println(s"size is ${arr.length}")
          println("first 10 elements: " + arr.take(10).map(_.toDouble.formatted("%.5e")).mkString(","))
          println(s"query: ${start.toDouble.formatted("%.5e")} - ${end.toDouble.formatted("%.5e")}")
      }
    }
    printSample(testGen.sample)
    printSample(testGen.sample)
    printSample(testGen.sample)
    val sample = testGen.sample.get
    val testArr = sample._1

    for (i <- 1 until 100) {
      println(s"testing $i")
      IndexBTree.write(testArr.take(i), hadoopCreate("/tmp/BTree.idx", sc.hadoopConfiguration))
    }
//    Spec.check
//    check(Spec)
//    forAll (testGen) {
//      arr: Array[Long] =>
//        val start = 0
//        val end = 5000000
//        val result = arr.flatMap(l => if (start <= l && end >= l) Some(l) else None)
//        IndexBTree.write(arr, path = FILENAME)
//        val query = IndexBTree.query(0, 50000, hadoopOpen(FILENAME, sc.hadoopConfiguration))
//        println(result.sameElements(query))
////        result.sameElements(query) ==> throws[RuntimeException]("something is wrong")
//    }
  }
}