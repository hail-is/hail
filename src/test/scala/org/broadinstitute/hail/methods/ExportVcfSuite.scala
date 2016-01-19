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

    // test that new VDS is same as old VDS
    println(vdsOrig.metadata == vdsNew.metadata)
    val rdd1 = vdsOrig.rdd.map{case (v,va,gs) => (v, gs)}.collect().toMap
    val rdd2 = vdsNew.rdd.map{case (v,va,gs) => (v, gs)}.collect().toMap
    val ann1 = vdsOrig.variantsAndAnnotations.collect().toMap
    val ann2 = vdsNew.variantsAndAnnotations.collect().toMap

    rdd1.foreach {
      case (v, gs) =>
        val gs2 = rdd2(v)
        if (!gs.sameElements(gs2)) {
          println(s"ERROR FOLLOWING ($v):")
          println(gs)
          println(gs2)
        }
    }
    ann1.foreach {
      case (v, va) =>
        val va2 = ann2(v)
        if (!(va == va2)) {
          println(s"ERROR FOLLOWING ($v):")
          println(va.attrs)
          println(va2.attrs)
        }
    }


//    println(vdsOrig.variants.collect().toIndexedSeq == vdsNew.variants.collect().toIndexedSeq)
//    println(vdsOrig.variants.collect().toSet == vdsNew.variants.collect().toSet)
//    println(rdd1(0))
//    println(rdd2(0))
//    rdd1.zip(rdd2).foreach {
//      case (a,b) =>
//        if (!(a == b))
//          println(a)
//          println(b)
//    }
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