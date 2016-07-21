package org.broadinstitute.hail.variant.vsm

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.driver.{Read, Repartition, State, Write}
import org.broadinstitute.hail.variant.{VSMSubgen, Variant, VariantSampleMatrix}
import org.testng.annotations.Test

class PartitioningSuite extends SparkSuite {

  @Test def testParquetWriteRead() {
    Prop.forAll(VariantSampleMatrix.gen(sc, VSMSubgen.random), Gen.choose(1, 10)) { case (vds, nPar) =>
      var state = State(sc, sqlContext, vds)
      state = Repartition.run(state, Array("-n", nPar.toString))
      val out = tmpDir.createTempFile("out", ".vds")
      val out2 = tmpDir.createTempFile("out", ".vds")
      state = Write.run(state, Array("-o", out))

      // need to do 2 writes to ensure that the RDD is ordered
      state = Read.run(state, Array("-i", out))
      state = Write.run(state, Array("-o", out2))
      val readback = Read.run(state, Array("-i", out2))


      state.vds.variantsAndAnnotations
        .zipPartitions(readback.vds.variantsAndAnnotations)(
          { (it1: Iterator[(Variant, Annotation)],
          it2: Iterator[(Variant, Annotation)]) => it1.zip(it2)
          })
        .collect()
        .foreach { case (t1, t2) =>
          assert(t1 == t2)
        }

      true
    }.check()
  }
}
