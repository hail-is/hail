package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.testng.annotations.Test

import scala.language.implicitConversions

class IndexBTreeSuite extends SparkSuite {

  object Spec extends Properties("BTree") {

    val arraySizeGenerator = for (depth: Int <- frequency[Int]((4,const(1)),(5,const(2)),(1,const(3)));
                              arraySize: Int <- choose(math.max(1, math.pow(10, (depth - 1) * math.log10(1024)).toInt),
                                math.min(1100000,math.pow(10, depth * math.log10(1024)).toInt))) yield (depth, arraySize)

    def fillRandomArray(arraySize: Int): Array[Long] = {
      val randArray = new Array[Long](arraySize)
      var pos = 24.toLong
      var nIndices = 0
      while (nIndices != arraySize) {
        randArray(nIndices) = pos
        pos += choose(5.toLong,5000.toLong).sample()
        nIndices += 1
      }
      randArray
    }

    property("query gives same answer as array") =
      forAll(arraySizeGenerator) { case (depth: Int, arraySize: Int) =>
        val arrayRandomStarts = fillRandomArray(arraySize)
        val maxLong = arrayRandomStarts.takeRight(1)(0)
        val index = "/tmp/testBtree.idx"

        hadoopDelete(index, sc.hadoopConfiguration, true)
        IndexBTree.write(arrayRandomStarts, index, sc.hadoopConfiguration)

        val indexSize = hadoopGetFileSize(index, sc.hadoopConfiguration)
        val padding = if (arraySize <= 1024) 1024 - arraySize else arraySize % 1024
        val numEntries = arraySize + padding + (1 until depth).map{math.pow(1024,_).toInt}.sum

        // make sure index size is correct
        val indexCorrectSize = if (indexSize == (numEntries * 8)) true else false

        // make sure depth is correct
        val estimatedDepth = IndexBTree.calcDepth(index, sc.hadoopConfiguration)
        val depthCorrect = if (estimatedDepth == depth) true else false

        // make sure query is correct
        val queryCorrect = if (arrayRandomStarts.length < 100)
          arrayRandomStarts.forall{case (l) => IndexBTree.queryIndex(l-1, maxLong + 5, index, sc.hadoopConfiguration) == l}
        else {
          val randomIndices = Array(0) ++ Array.fill(100)(choose(0,arraySize - 1).sample())
          randomIndices.map(arrayRandomStarts).forall { case (l) => IndexBTree.queryIndex(l-1,maxLong + 5, index, sc.hadoopConfiguration) == l }
        }

        depthCorrect && indexCorrectSize && queryCorrect
      }
  }

  @Test def test() {
    Spec.check()
  }

  @Test def oneVariant() {
    val index = Array(24.toLong)
    val fileSize = 30 //made-up value greater than index
    val idxFile = "/tmp/testBtree_1variant.idx"
    val hConf = sc.hadoopConfiguration

    hadoopDelete(idxFile, sc.hadoopConfiguration, true)
    IndexBTree.write(index, idxFile, hConf)

    intercept[FatalException]{
      IndexBTree.queryIndex(-5, fileSize, idxFile, hConf)
    }

    intercept[FatalException]{
      IndexBTree.queryIndex(100, fileSize, idxFile, hConf)
    }

    assert(IndexBTree.queryIndex(0, fileSize, idxFile, hConf) == 24)
    assert(IndexBTree.queryIndex(10, fileSize, idxFile, hConf) == 24)
    assert(IndexBTree.queryIndex(20, fileSize, idxFile, hConf) == 24)
    assert(IndexBTree.queryIndex(24, fileSize, idxFile, hConf) == 24)
    assert(IndexBTree.queryIndex(25, fileSize, idxFile, hConf) == -1)
    assert(IndexBTree.queryIndex(fileSize - 1, fileSize, idxFile, hConf) == -1)
  }

  @Test def zeroVariants() {
    intercept[IllegalArgumentException] {
      val index = Array[Long]()
      val fileSize = 30 //made-up value greater than index
      val idxFile = "/tmp/testBtree_0variant.idx"
      hadoopDelete(idxFile, sc.hadoopConfiguration, true)
      IndexBTree.write(index, idxFile, sc.hadoopConfiguration)
    }
  }
}