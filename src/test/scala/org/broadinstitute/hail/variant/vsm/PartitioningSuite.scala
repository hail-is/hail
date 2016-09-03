package org.broadinstitute.hail.variant.vsm

import org.broadinstitute.hail.{PropertySuite, SparkSuite}
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.driver.{Coalesce, Read, State, Write}
import org.broadinstitute.hail.variant.{VSMSubgen, VariantSampleMatrix}

class PartitioningProperties extends PropertySuite {

  property("parquet read write") = forAll(VariantSampleMatrix.gen(sc, VSMSubgen.random), Gen.choose(1, 10)) { case (vds, nPar) =>
    var state = State(sc, sqlContext, vds)
    state = Coalesce.run(state, Array("-n", nPar.toString))
    val out = tmpDir.createTempFile("out", ".vds")
    val out2 = tmpDir.createTempFile("out", ".vds")
    state = Write.run(state, Array("-o", out))

    // need to do 2 writes to ensure that the RDD is ordered
    state = Read.run(state, Array("-i", out))
    state = Write.run(state, Array("-o", out2))
    val readback = Read.run(state, Array("-i", out2))

    val original = state.vds.variantsAndAnnotations
      .mapPartitionsWithIndex { case (i, it) => it.zipWithIndex.map(x => (i, x)) }
      .collect()
      .toSet
    val rb = readback.vds.variantsAndAnnotations
      .mapPartitionsWithIndex { case (i, it) => it.zipWithIndex.map(x => (i, x)) }
      .collect()
      .toSet
    rb == original
  }
}
