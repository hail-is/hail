package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test
import scala.io.Source

class ExportVcfSuite extends SparkSuite {
  @Test def testSameAsOrig() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val tmpDir = "/tmp/"
    val outFile = tmpDir + "testExportVcf.vcf"

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State("", sc, sqlContext, vdsOrig)
    ExportVCF.run(stateOrig, Array("-o", outFile, "-t", tmpDir))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))
    val stateNew = State("", sc, sqlContext, vdsNew)

    // test that new VDS is same as old VDS
    assert(stateOrig.vds.same(stateNew.vds))
  }

  @Test def testSorted() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val tmpDir = "/tmp/"
    val outFile = tmpDir + "testExportVcf.vcf"

    val vdsOrig = LoadVCF(sc, vcfFile, nPartitions = Some(10))
    val stateOrig = State("", sc, sqlContext, vdsOrig)
    ExportVCF.run(stateOrig, Array("-o", outFile, "-t", tmpDir))

    val vdsNew = LoadVCF(sc, outFile, nPartitions = Some(10))
    val stateNew = State("", sc, sqlContext, vdsNew)

    case class Coordinate(contig:String,start:Int,ref:String,alt:String) extends Ordered[Coordinate] {
      def compare(that:Coordinate) = {
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

    // check output is sorted
    val coordinates:Array[Coordinate] = Source.fromFile(outFile).getLines().filter(line => !line.isEmpty && line(0) != '#').map(line => line.split("\t")).take(5).map(a => new Coordinate(a(0),a(1).toInt,a(3),a(4))).toArray
    val sortedCoordinates = coordinates.sorted

    assert(sortedCoordinates.sameElements(coordinates))

  }
}
