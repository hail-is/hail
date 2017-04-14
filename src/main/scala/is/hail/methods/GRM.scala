package is.hail.methods

import java.io.DataOutputStream

import breeze.linalg.SparseVector
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

object GRM {
  def writeFloatLittleEndian(s: DataOutputStream, f: Float) {
    val bits: Int = java.lang.Float.floatToRawIntBits(f)
    s.write(bits & 0xff)
    s.write((bits >> 8) & 0xff)
    s.write((bits >> 16) & 0xff)
    s.write(bits >> 24)
  }

  def apply(vds: VariantDataset): KinshipMatrix = {
    /*if (!Set("rel", "gcta-grm-bin", "gcta-grm").contains(format))
      fatal(s"unknown format `$format', expect one of `rel', `gcta-grm', `gcta-grm-bin'")*/

    val (_, mat) = ToHWENormalizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == mat.numCols())
    val nVariants = mat.numRows() // mat cached

    val bmat = mat.toBlockMatrixDense().cache()
    val grm = bmat.transpose.multiply(bmat)

    assert(grm.numCols == nSamples
      && grm.numRows == nSamples)

    new KinshipMatrix(grm.toIndexedRowMatrix, vds.sampleIds.toArray)
  }
}