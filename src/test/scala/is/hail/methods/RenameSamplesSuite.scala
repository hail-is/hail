package is.hail.methods

import is.hail.SparkSuite
import is.hail.utils._
import is.hail.check.Gen
import is.hail.driver.{ImportVCF, RenameSamples, State}
import is.hail.utils.FatalException
import org.testng.annotations.Test

import scala.collection.mutable

class RenameSamplesSuite extends SparkSuite {

  def writeSampleMap(file: String, m: mutable.Map[String, String]) {
    hadoopConf.writeTable(file, m.map { case (k, v) => s"$k\t$v" })
  }

  @Test def testCollision() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    val m = mutable.Map[String, String](
      s.vds.sampleIds(0) -> "a",
      s.vds.sampleIds(1) -> "a"
    )

    val samplesMapFile = tmpDir.createTempFile("samples", extension = ".map")

    writeSampleMap(samplesMapFile, m)

    // FIXME keyword
    intercept[FatalException] {
      s = RenameSamples.run(s, Array("-i", samplesMapFile))
    }
  }

  @Test def test() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    val samples = s.vds.sampleIds.toSet
    val newSamples = mutable.Set[String](samples.toArray: _*)
    val m = mutable.Map.empty[String, String]
    val genNewSample = Gen.identifier.filter(newS => !newSamples.contains(newS))
    for (s <- s.vds.sampleIds) {
      Gen.option(genNewSample, 0.5).sample() match {
        case Some(newS) =>
          newSamples -= s
          newSamples += newS
          m += s -> newS
        case None =>
      }
    }

    // add a few extraneous entries whose domain don't overlap with samples
    for (i <- 1 to 10) {
      val newS = Gen.identifier.filter(s => !samples.contains(s)).sample()
      val s = Gen.identifier.filter(s => !newSamples.contains(s)).sample()
      m += newS -> s
    }

    val samplesMapFile = tmpDir.createTempFile("samples", extension = ".map")

    writeSampleMap(samplesMapFile, m)

    s = RenameSamples.run(s, Array("-i", samplesMapFile))

    assert(s.vds.sampleIds.toSet == newSamples)
  }
}
