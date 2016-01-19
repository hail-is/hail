package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test
import scala.io.Source

class ExportVcfSuite extends SparkSuite {

  @Test def testSameAsOrigBGzip() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = "/tmp/testExportVcf.vcf.bgz"

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State(sc, sqlContext, vdsOrig)

    ExportVCF.run(stateOrig, Array("-o", outFile))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))

    // test that new VDS is same as old VDS
    assert(vdsOrig.same(vdsNew))
  }

  @Test def testSameAsOrigNoCompression() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = "/tmp/testExportVcf.vcf"

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State(sc, sqlContext, vdsOrig)

    ExportVCF.run(stateOrig, Array("-o", outFile))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))

    assert(vdsOrig.same(vdsNew))
  }

  @Test def testSorted() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = "/tmp/testSortVcf.vcf.bgz"

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
}