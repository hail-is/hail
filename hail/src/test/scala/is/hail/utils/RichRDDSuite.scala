package is.hail.utils

import is.hail.{HailSuite, TestUtils}
import org.testng.annotations.Test

class RichRDDSuite extends HailSuite {
  @Test def parallelWrite() {
    def read(file: String): Array[String] = hc.sFS.readLines(file)(_.map(_.value).toArray)

    val header = "my header is awesome!"
    val data = Array("the cat jumped over the moon.", "all creatures great and small")
    val r = sc.parallelize(data, numSlices = 2)
    assert(r.getNumPartitions == 2)

    val concatenated = tmpDir.createTempFile("concatenated")
    r.writeTable(sFS, concatenated, tmpDir.createTempFile("concatenated"), Some(header), exportType = ExportType.CONCATENATED)

    assert(read(concatenated) sameElements (header +: data))

    val shardHeaders = tmpDir.createTempFile("shardHeader")
    r.writeTable(sFS, shardHeaders, tmpDir.createTempFile("shardHeader"), Some(header), exportType = ExportType.PARALLEL_HEADER_IN_SHARD)

    assert(read(shardHeaders + "/part-00000") sameElements header +: Array(data(0)))
    assert(read(shardHeaders + "/part-00001") sameElements header +: Array(data(1)))

    val separateHeader = tmpDir.createTempFile("separateHeader", ".gz")
    r.writeTable(sFS, separateHeader, tmpDir.createTempFile("separateHeader"), Some(header), exportType = ExportType.PARALLEL_SEPARATE_HEADER)

    assert(read(separateHeader + "/header.gz") sameElements Array(header))
    assert(read(separateHeader + "/part-00000.gz") sameElements Array(data(0)))
    assert(read(separateHeader + "/part-00001.gz") sameElements Array(data(1)))


    val merged = tmpDir.createTempFile("merged", ".gz")
    val mergeList = Array(separateHeader + "/header.gz",
      separateHeader + "/part-00000.gz",
      separateHeader + "/part-00001.gz").flatMap(sFS.glob)
    sFS.copyMergeList(mergeList, merged, deleteSource = false)

    assert(read(merged) sameElements read(concatenated))
  }
}
