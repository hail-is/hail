package org.broadinstitute.hail.methods

import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.driver.{RenameSamples, ImportVCF, State}
import org.testng.annotations.Test
import scala.collection.mutable

class RenameSamplesSuite extends SparkSuite {

  def writeSampleMap(file: String, m: mutable.Map[String, String]) {
    writeTable(file, sc.hadoopConfiguration, m.map { case (k, v) => s"$k\t$v" })
  }

  @Test def testCollision() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    val m = mutable.Map[String, String](
      s.vds.sampleIds(0) -> "a",
      s.vds.sampleIds(1) -> "a"
    )

    writeSampleMap("/tmp/samples.map", m)

    // FIXME keyword
    intercept[FatalException] {
      s = RenameSamples.run(s, Array("-i", "/tmp/samples.map"))
    }
  }

  @Test def test() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    val samples = s.vds.sampleIds.toSet
    val newSamples = mutable.Set[String](samples.toArray: _*)
    val m = mutable.Map.empty[String, String]
    val genNewSample = Gen.identifier.filter(newS => !samples.contains(newS))
    for (s <- s.vds.sampleIds) {
      Gen.option(genNewSample, 0.5).sample() match {
        case Some(newS) =>
          newSamples -= s
          newSamples += newS
          m += s -> newS
        case None =>
      }
    }

    for (i <- 1 to 10) {
      val newS = genNewSample.sample()
      val s = Gen.identifier.sample()
      m += newS -> s
    }

    writeSampleMap("/tmp/samples.map", m)

    s = RenameSamples.run(s, Array("-i", "/tmp/samples.map"))

    assert(s.vds.sampleIds.toSet == newSamples)
  }
}
