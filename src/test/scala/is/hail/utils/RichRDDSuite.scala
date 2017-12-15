package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class RichRDDSuite extends SparkSuite {
  @Test def testTakeByPartition() {
    val r = sc.parallelize(0 until 1024, numSlices = 20)
    assert(r.headPerPartition(5).count() == 100)
  }

  @Test def testHead() {
    val r = sc.parallelize(0 until 1024, numSlices = 20)
    val partitionRanges = r.countPerPartition().scanLeft(Range(0, 0)) { case (x, c) => Range(x.end, x.end + c.toInt) }

    def getExpectedNumPartitions(n: Int): Int =
      partitionRanges.indexWhere(_.contains(math.max(0, n - 1)))

    for (n <- Array(0, 15, 200, 562, 1024, 2000)) {
      val t = r.head(n)
      val nActual = math.min(n, 1024)

      assert(t.collect() sameElements (0 until nActual))
      assert(t.count() == nActual)
      assert(t.getNumPartitions == getExpectedNumPartitions(nActual))
    }

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    assert(vds.head(3).countVariants() == 3)
  }

  @Test def binaryParallelWrite() {
    def readBytes(file: String): Array[Byte] = hadoopConf.readFile(file) { dis =>
      val buffer = new Array[Byte](32)
      val size = dis.read(buffer)
      buffer.take(size)
    }

    val header = Array[Byte](108, 27, 1, 91)
    val data = Array(Array[Byte](1, 19, 23, 127, -1), Array[Byte](23, 4, 15, -2, 1))
    val r = sc.parallelize(data, numSlices = 2)
    assert(r.getNumPartitions == 2)

    val notParallelWrite = tmpDir.createTempFile("notParallelWrite")
    r.saveFromByteArrays(notParallelWrite, tmpDir.createTempFile("notParallelWrite_tmp"), Some(header), exportType = ExportType.CONCATENATED)

    assert(readBytes(notParallelWrite) sameElements (header ++: data.flatten))

    val parallelWrite = tmpDir.createTempFile("parallelWrite")
    r.saveFromByteArrays(parallelWrite, tmpDir.createTempFile("parallelWrite_tmp"), Some(header), exportType = ExportType.PARALLEL_HEADER_IN_SHARD)

    assert(readBytes(parallelWrite + "/part-00000") sameElements header ++ data(0))
    assert(readBytes(parallelWrite + "/part-00001") sameElements header ++ data(1))

    val parallelWriteHeader = tmpDir.createTempFile("parallelWriteHeader")
    r.saveFromByteArrays(parallelWriteHeader, tmpDir.createTempFile("parallelHeaderWrite_tmp"), Some(header), exportType = ExportType.PARALLEL_SEPARATE_HEADER)

    assert(readBytes(parallelWriteHeader + "/header") sameElements header)
    assert(readBytes(parallelWriteHeader + "/part-00000") sameElements data(0))
    assert(readBytes(parallelWriteHeader + "/part-00001") sameElements data(1))
  }
}
