package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.check.{Gen, Prop}
import is.hail.variant.{VSMSubgen, VariantSampleMatrix}
import org.testng.annotations.Test

class PartitioningSuite extends SparkSuite {

  @Test def testParquetWriteRead() {
    Prop.forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random), Gen.choose(1, 10)) { case (vds, nPar) =>

      val out = tmpDir.createTempFile("out", ".vds")
      val out2 = tmpDir.createTempFile("out", ".vds")

      vds.coalesce(nPar)
        .write(out)

      // need to do 2 writes to ensure that the RDD is ordered
      hc.read(out)
        .write(out2)

      val readback = hc.read(out2)

      val original = vds.variantsAndAnnotations
        .mapPartitionsWithIndex { case (i, it) => it.zipWithIndex.map(x => (i, x)) }
        .collect()
        .toSet
      val rb = readback.variantsAndAnnotations
        .mapPartitionsWithIndex { case (i, it) => it.zipWithIndex.map(x => (i, x)) }
        .collect()
        .toSet
      rb == original
    }.check()
  }
}
