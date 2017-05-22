package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.utils.{HailException, _}
import org.testng.annotations.Test

import scala.collection.mutable

class RenameSamplesSuite extends SparkSuite {

  def writeSampleMap(file: String, m: mutable.Map[String, String]) {
    hadoopConf.writeTable(file, m.map { case (k, v) => s"$k\t$v" })
  }

  @Test def testCollision() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val m = mutable.Map[Annotation, String](
      vds.sampleIds(0) -> "a",
      vds.sampleIds(1) -> "a"
    )

    // FIXME keyword
    intercept[HailException] {
      vds.renameSamples(m.toMap)
    }
  }

  @Test def test() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val samples = vds.sampleIds.toSet
    val newSamples = mutable.Set[Annotation](samples.toArray: _*)
    val m = mutable.Map.empty[Annotation, Annotation]
    val genNewSample = Gen.identifier.filter(newS => !newSamples.contains(newS))
    for (s <- vds.sampleIds) {
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

    val vds2 = vds.renameSamples(m.toMap)

    assert(vds2.sampleIds.toSet == newSamples)
  }
}
