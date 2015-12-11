package org.broadinstitute.hail.io

import org.broadinstitute.hail.SparkSuite
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils.hadoopOpen

class IndexBTreeSuite extends SparkSuite {
  @Test def test() {
    val hadoopConf = sc.hadoopConfiguration


    val a1 = (0 until 500).map{ i => (i*10000).toLong}.toArray
    val a2 = (0 until 500000).map{ i => (i*100).toLong}.toArray
    val a3 = (0 until 5000000).map{ i => i.toLong}.toArray
    val a1w = IndexBTree.write(a1, "src/test/resources/500Array.idx")
    val a2w = IndexBTree.write(a2, "src/test/resources/500KArray.idx")
//    val a3w = IndexBTree.write(a3, "src/test/resources/5MArray.idx")

    println
    println("about to read 500k")

    println("ret is: " + IndexBTree.query(200000, 205010, hadoopOpen("src/test/resources/500KArray.idx", hadoopConf)).mkString(","))
//    println(IndexBTree.queryArr(400050, 400500, a2w).mkString(","))
//    println(IndexBTree.queryArr(400050, 400095, a3w).mkString(","))
  }
}