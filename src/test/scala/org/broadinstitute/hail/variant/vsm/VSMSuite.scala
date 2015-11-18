package org.broadinstitute.hail.variant.vsm

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.variant.{Variant, VariantSampleMatrix}
import org.broadinstitute.hail.Utils._
import scala.collection.mutable
import scala.util.Random
import sys.process._
import scala.language.postfixOps
import org.broadinstitute.hail.methods.{sSingletonVariants, LoadVCF}
import org.testng.annotations.Test

class VSMSuite extends SparkSuite {
  // FIXME
  val vsmTypes = List("sparky")

  @Test def testsSingletonVariants() {
    val singletons: List[Set[Variant]] =
      vsmTypes
        .map(vsmtype => {
          val vdsdir = "/tmp/sample." + vsmtype + ".vds"

          val result = "rm -rf " + vdsdir !;
          assert(result == 0)

          LoadVCF(sc, "src/test/resources/sample.vcf.gz", vsmtype = vsmtype)
            .write(sqlContext, vdsdir)

          val vds = VariantSampleMatrix.read(sqlContext, vdsdir)
          sSingletonVariants(vds)
        })

    assert(singletons.tail.forall(s => s == singletons.head))
  }

  @Test def testFilterSamples() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf.gz")
    val vdsAsMap = vds.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
    val nSamples = vds.nSamples
    assert(nSamples == vds.nLocalSamples)

    // FIXME ScalaCheck
    for (n <- 0 until 20) {
      val keep = mutable.Set.empty[Int]

      // n == 0: none
      if (n == 1) {
        for (i <- 0 until nSamples)
          keep += i
      } else if (n > 1) {
        for (i <- 0 until nSamples) {
          if (Random.nextFloat() < 0.5)
            keep += i
        }
      }

      vsmTypes.foreach { vsmtype =>
        val localKeep = keep
        val filtered = LoadVCF(sc, "src/test/resources/sample.vcf.gz", vsmtype = vsmtype)
          .filterSamples(s => localKeep(s))

        val filteredAsMap = filtered.mapWithKeys((v, s, g) => ((v, s), g)).collectAsMap()
        filteredAsMap.foreach { case (k, g) => simpleAssert(vdsAsMap(k) == g) }

        simpleAssert(filtered.nSamples == nSamples)
        simpleAssert(filtered.localSamples.toSet == keep)

        val sampleKeys = filtered.mapWithKeys((v, s, g) => s).distinct.collect()
        assert(sampleKeys.toSet == keep)
      }
    }
  }
}
