package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr.TStruct
import org.broadinstitute.hail.variant.{Genotype, VariantSampleMatrix}
import org.testng.annotations.Test

import scala.io.Source

class ExportVcfSuite extends SparkSuite {

  @Test def testSameAsOrigBGzip() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", ".vcf")

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State(sc, sqlContext, vdsOrig)

    ExportVCF.run(stateOrig, Array("-o", outFile))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))

    assert(vdsOrig.same(vdsNew))
  }

  @Test def testSameAsOrigNoCompression() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", ".vcf")
    val outFile2 = tmpDir.createTempFile("export2", ".vcf")

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State(sc, sqlContext, vdsOrig)

    ExportVCF.run(stateOrig, Array("-o", outFile))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))

    assert(vdsOrig.eraseSplit.same(vdsNew.eraseSplit))

    val infoSize = vdsNew.vaSignature.getAsOption[TStruct]("info").get.size
    val toAdd = Some(Annotation.fromSeq(Array.fill[Any](infoSize)(null)))
    val (_, inserter) = vdsNew.insertVA(null, "info")

    val vdsNewMissingInfo = vdsNew.mapAnnotations((v, va) => inserter(va, toAdd))

    ExportVCF.run(stateOrig.copy(vds = vdsNewMissingInfo), Array("-o", outFile2))

    assert(LoadVCF(sc, outFile2).eraseSplit.same(vdsNewMissingInfo.eraseSplit))
  }

  @Test def testSorted() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("sort", ".vcf.bgz")

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State(sc, sqlContext, vdsOrig)

    ExportVCF.run(stateOrig, Array("-o", outFile))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))
    val stateNew = State(sc, sqlContext, vdsNew)

    case class Coordinate(contig: String, start: Int, ref: String, alt: String) extends Ordered[Coordinate] {
      def compare(that: Coordinate) = {
        if (this.contig != that.contig)
          this.contig.compareTo(that.contig)
        else if (this.start != that.start)
          this.start.compareTo(that.start)
        else if (this.ref != that.ref)
          this.ref.compareTo(that.ref)
        else
          this.alt.compareTo(that.alt)
      }
    }
    val coordinates: Array[Coordinate] = readFile(outFile, stateNew.hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty && line(0) != '#')
        .map(line => line.split("\t")).take(5).map(a => new Coordinate(a(0), a(1).toInt, a(3), a(4))).toArray
    }

    val sortedCoordinates = coordinates.sortWith { case (c1, c2) => c1.compare(c2) < 0 }

    assert(sortedCoordinates.sameElements(coordinates))
  }

  @Test def testReadWrite() {
    val s = State(sc, sqlContext, null)
    val p = forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)) { (vsm: VariantSampleMatrix[Genotype]) =>
      hadoopDelete("/tmp/foo.vcf", sc.hadoopConfiguration, recursive = true)
      ExportVCF.run(s.copy(vds = vsm), Array("-o", "/tmp/foo.vcf"))
      val vsm2 = ImportVCF.run(s, Array("/tmp/foo.vcf")).vds
      ExportVCF.run(s.copy(vds = vsm2), Array("-o", "/tmp/foo2.vcf"))
      val vsm3 = ImportVCF.run(s, Array("/tmp/foo2.vcf")).vds
      vsm2.same(vsm3)
    }

      p.check
  }
}