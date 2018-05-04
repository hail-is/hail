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
    val partitionRanges = r.countPerPartition().scanLeft(Range(0, 1)) { case (x, c) => Range(x.end, x.end + c.toInt + 1) }

    def getExpectedNumPartitions(n: Int): Int =
      partitionRanges.indexWhere(_.contains(n))

    for (n <- Array(0, 15, 200, 562, 1024, 2000)) {
      val t = r.head(n)
      val nActual = math.min(n, 1024)

      assert(t.collect() sameElements (0 until nActual))
      assert(t.count() == nActual)
      assert(t.getNumPartitions == getExpectedNumPartitions(nActual))
    }

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    assert(vds.head(3).countRows() == 3)
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

  @Test def parallelWrite() {
    def read(file: String): Array[String] = hc.hadoopConf.readLines(file)(_.map(_.value).toArray)

    val header = "my header is awesome!"
    val data = Array("the cat jumped over the moon.", "all creatures great and small")
    val r = sc.parallelize(data, numSlices = 2)
    assert(r.getNumPartitions == 2)

    val concatenated = tmpDir.createTempFile("concatenated")
    r.writeTable(concatenated, tmpDir.createTempFile("concatenated"), Some(header), exportType = ExportType.CONCATENATED)

    assert(read(concatenated) sameElements (header +: data))

    val shardHeaders = tmpDir.createTempFile("shardHeader")
    r.writeTable(shardHeaders, tmpDir.createTempFile("shardHeader"), Some(header), exportType = ExportType.PARALLEL_HEADER_IN_SHARD)

    assert(read(shardHeaders + "/part-00000") sameElements header +: Array(data(0)))
    assert(read(shardHeaders + "/part-00001") sameElements header +: Array(data(1)))

    val separateHeader = tmpDir.createTempFile("separateHeader", ".gz")
    r.writeTable(separateHeader, tmpDir.createTempFile("separateHeader"), Some(header), exportType = ExportType.PARALLEL_SEPARATE_HEADER)

    assert(read(separateHeader + "/header.gz") sameElements Array(header))
    assert(read(separateHeader + "/part-00000.gz") sameElements Array(data(0)))
    assert(read(separateHeader + "/part-00001.gz") sameElements Array(data(1)))


    val merged = tmpDir.createTempFile("merged", ".gz")
    val mergeList = Array(separateHeader + "/header.gz",
      separateHeader + "/part-00000.gz",
      separateHeader + "/part-00001.gz").flatMap(hadoopConf.glob)
    hadoopConf.copyMergeList(mergeList, merged, deleteSource = false)

    assert(read(merged) sameElements read(concatenated))
  }
}
