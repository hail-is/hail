package is.hail.methods

import is.hail.utils._

import scala.io.Source
import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.TString
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.testng.annotations.Test

/**
  * Tests for KinshipMatrix. See GRMSuite for tests relevant to other export methods.
  */
class KinshipMatrixSuite extends SparkSuite {

  val data = Seq(
    IndexedRow(1L, Vectors.dense(5, 6, 7, 8)),
    IndexedRow(2L, Vectors.dense(9, 10, 11, 12)),
    IndexedRow(3L, Vectors.dense(13, 14, 15, 16))
  )


  @Test def testFilterSamplesDimensions() {
    val km = hc.baldingNicholsModel(1, 15, 15).rrm()
    val kmFilt = km.filterSamples { s =>
      val n = s.asInstanceOf[String].toInt
      n < 7 && n > 3
    }
    val kmFiltLocalMatrix = kmFilt.matrix.toBlockMatrix().toLocalMatrix()

    assert(kmFilt.sampleIds.length == 3)
    assert(kmFiltLocalMatrix.numCols == 3)
    assert(kmFiltLocalMatrix.numRows == 3)
  }

  @Test def testFilterSamplesValues() {
    val irdd = sc.parallelize(data)
    val irm = new IndexedRowMatrix(irdd)
    val samples = (0 to 3).map(i => s"S$i")
    val km = KinshipMatrix(hc, TString, irm, samples.toArray, 10)

    val kmOneEntry = km.filterSamples(s => s == "S2")
    assert(kmOneEntry.matrix.toBlockMatrix().toLocalMatrix()(0, 0) == 11)

    val kmTwoByTwo = km.filterSamples(s => s == "S1" || s == "S3")
    assert(kmTwoByTwo.matrix.toBlockMatrix().toLocalMatrix().toArray.toSeq == Seq(6.0, 14.0, 8.0, 16.0))
  }

  @Test def exportTSV() {
    val irdd = sc.parallelize(data)
    val irm = new IndexedRowMatrix(irdd)
    val samples = (0 to 3).map(i => s"S$i")
    val km = KinshipMatrix(hc, TString, irm, samples.toArray, 10)

    val out = tmpDir.createTempFile("kinshipMatrixExportTSVTest", ".tsv")

    km.exportTSV(out)

    val readInValues = hadoopConf.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .drop(1)
        .map(s => s.split("\t").map(_.toDouble)).toArray
    }

    //Ensures that all 4 rows of data matrix got written out and read back in, even though row 0 is not explicity included in data.
    assert(readInValues.length == irm.numRows())

    assert(km.matrix.toBlockMatrix().toLocalMatrix().rowIter.map(v => v.toArray).toArray.deep == readInValues.deep)

  }
}
