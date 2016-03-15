package org.broadinstitute.hail.io

import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.Utils._
import org.testng.annotations.Test
import scala.language.implicitConversions
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Gen._

class IndexBTreeSuite extends SparkSuite {

  object Spec extends Properties("BTree") {

    val arraySizeGenerator = for (depth: Int <- choose(1,2);
                              arraySize: Int <- choose(math.max(1, math.pow(10, (depth - 1) * math.log10(1024)).toInt),
                                math.pow(10, depth * math.log10(1024)).toInt)) yield (depth, arraySize)

    def fillRandomArray(arraySize: Int): Array[Long] = {
      val randArray = new Array[Long](arraySize)
      var pos = 24.toLong
      var nIndices = 0
      while (nIndices != arraySize) {
        randArray(nIndices) = pos
        pos += choose(1.toLong,5000.toLong).sample()
        nIndices += 1
      }
      randArray
    }

    property("index is correct size after write") =
      forAll(arraySizeGenerator) { case (depth: Int, arraySize: Int) =>
        val arrayRandomStarts = fillRandomArray(arraySize)
        IndexBTree.write(arrayRandomStarts, "/tmp/testBtree.idx", sc.hadoopConfiguration)
        val indexSize = hadoopGetFileSize("/tmp/testBtree.idx", sc.hadoopConfiguration)
        val depth = math.max(1,(math.log10(arrayRandomStarts.length) / math.log10(1024)).ceil.toInt)
        val numEntries = arraySize + (0 until depth).map{math.pow(1024,_).toInt}.sum

        if (indexSize == (numEntries * 8)) {
          true
        }
        else {
          false
        }
      }

    property("query gives same answer as array") =
      forAll(arraySizeGenerator) { case (depth: Int, arraySize: Int) =>
        val arrayRandomStarts = fillRandomArray(arraySize)
        //println(s"arraysize=$arraySize depth=$depth")

        IndexBTree.write(arrayRandomStarts, "/tmp/testBtree.idx", sc.hadoopConfiguration)
        val indexsize = hadoopGetFileSize("/tmp/testBtree.idx", sc.hadoopConfiguration)


        if (arrayRandomStarts.length < 0)
          arrayRandomStarts.forall{case (l) => IndexBTree.query(l,"/tmp/testBtree.idx", sc.hadoopConfiguration) == l}
        else {
          val randomIndices = Array.fill(100)(choose(0,arraySize - 1).sample())
          val maxLong = arrayRandomStarts.takeRight(1)(0)
          println(s"arraysize=$arraySize depth=$depth maxStart=$maxLong")
          //randomIndices.map(arrayRandomStarts).forall{case (l) => l <= maxLong && l >= 0.toLong}
          randomIndices.map(arrayRandomStarts).forall { case (l) => IndexBTree.query(l, "/tmp/testBtree.idx", sc.hadoopConfiguration) == l }
        }
      }
  }

/*  @Test def test() {
    Spec.check()
  }*/

  @Test def oneVariant() {
    val index = Array(24.toLong)
    IndexBTree.write(index, "/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)
    index.forall{case (l) => IndexBTree.query(l,"/tmp/testBtree.idx", sc.hadoopConfiguration) == l}
  }

  @Test def zeroVariants() {
    intercept[IllegalArgumentException] {
      val index = Array[Long]()
      IndexBTree.write(index, "/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)
      index.forall { case (l) => IndexBTree.query(l, "/tmp/testBtree.idx", sc.hadoopConfiguration) == l }
    }
  }

  @Test def jackieTest() {
    val index = Array(24.toLong)
    val idx = "/tmp/testBtree_1variant.idx"
    val hConf = sc.hadoopConfiguration
    IndexBTree.write(index, "/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)
    intercept[FatalException]{
      IndexBTree.queryJackie(-5,idx,hConf)
    }
    assert(IndexBTree.queryJackie(0,idx,hConf) == 24)
    assert(IndexBTree.queryJackie(10,idx,hConf) == 24)
    assert(IndexBTree.queryJackie(20,idx,hConf) == 24)
    assert(IndexBTree.queryJackie(24,idx,hConf) == 24)
    assert(IndexBTree.queryJackie(25,idx,hConf) == -1)
    assert(IndexBTree.queryJackie(30,idx,hConf) == -1)
    assert(IndexBTree.queryJackie(100,idx,hConf) == -1)

    println(s"0:${IndexBTree.queryJackie(0,"/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)}")
    println(s"10:${IndexBTree.queryJackie(10,"/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)}")
    println(s"20:${IndexBTree.queryJackie(20,"/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)}")
    println(s"24:${IndexBTree.queryJackie(24,"/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)}")
    println(s"30:${IndexBTree.queryJackie(30,"/tmp/testBtree_1variant.idx", sc.hadoopConfiguration)}")

    index.forall{case (l) => IndexBTree.query(l-1,"/tmp/testBtree_1variant.idx", sc.hadoopConfiguration) == l}
  }

  // assert index < 1024
  // seek function of where to go next == total size of all layers up to one you're on plus index times 8KB (1024 * 8)
}