package is.hail.methods

import is.hail.utils._
import scala.io.Source
import is.hail.SparkSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.testng.annotations.Test

/**
  * Tests for KinshipMatrix.
  */
class KinshipMatrixSuite extends SparkSuite {

  val data = Seq(
    IndexedRow(0L, Vectors.dense(1, 2, 3, 4)),
    IndexedRow(1L, Vectors.dense(5, 6, 7, 8)),
    IndexedRow(2L, Vectors.dense(9, 10, 11, 12)),
    IndexedRow(3L, Vectors.dense(13, 14, 15, 16))
  )


  @Test def testFilterSamplesDimensions() {
    val km = hc.baldingNicholsModel(1, 15, 15).rrm()
    val kmFilt = km.filterSamples(s => s.toInt < 7 && s.toInt > 3)
    val kmFiltLocalMatrix = kmFilt.matrix.toBlockMatrix().toLocalMatrix()

    assert(kmFilt.sampleIds.length == 3)
    assert(kmFiltLocalMatrix.numCols == 3)
    assert(kmFiltLocalMatrix.numRows == 3)
  }

  @Test def testFilterSamplesValues() {
    val irdd = sc.parallelize(data)
    val irm = new IndexedRowMatrix(irdd)
    val samples = (0 to 3).map(i => s"S$i")
    val km = new KinshipMatrix(hc, irm, samples.toArray)

    val kmOneEntry = km.filterSamples(Set("S2").contains)
    assert(kmOneEntry.matrix.toBlockMatrix().toLocalMatrix()(0, 0) == 11)

    val kmTwoByTwo = km.filterSamples(Set("S1", "S3").contains)
    assert(kmTwoByTwo.matrix.toBlockMatrix().toLocalMatrix().toArray.toSeq == Seq(6.0, 14.0, 8.0, 16.0))
  }

  @Test def exportTSV() {
    val irdd = sc.parallelize(data)
    val irm = new IndexedRowMatrix(irdd)
    val samples = (0 to 3).map(i => s"S$i")
    val km = new KinshipMatrix(hc, irm, samples.toArray)

    val out = tmpDir.createTempFile("kinshipMatrixExportTSVTest", ".tsv")

    km.exportTSV(out)

    val readInValues = hadoopConf.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .drop(1)
        .map(s => s.split("\t").map(_.toDouble)).toArray
    }

    assert(km.matrix.toBlockMatrix().toLocalMatrix().rowIter.map(v => v.toArray).toArray.deep == readInValues.deep)

  }
}
