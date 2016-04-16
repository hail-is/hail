package org.broadinstitute.hail.driver

import java.io.{DataInputStream, File}

import breeze.linalg.DenseMatrix
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{Genotype, Variant, VariantSampleMatrix}
import org.testng.annotations.Test

import scala.io.Source
import sys.process._
import scala.language.postfixOps

class GRMSuite extends SparkSuite {
  def loadIDFile(file: String): Array[String] = {
    readFile(file, sc.hadoopConfiguration) { s =>
      Source.fromInputStream(s)
        .getLines()
        .map { line =>
          val s = line.split("\t")
          assert(s.length == 2)
          s(1)
        }.toArray
    }
  }

  def loadRel(nSamples: Int, file: String): DenseMatrix[Float] = {
    val m: DenseMatrix[Float] = new DenseMatrix[Float](rows = nSamples, cols = nSamples)
    val rows = readFile(file, sc.hadoopConfiguration) { s =>
      val lines = Source.fromInputStream(s)
        .getLines()
        .toArray
      assert(lines.length == nSamples)

      lines
        .zipWithIndex
        .foreach { case (line, i) =>
          val row = line.split("\t").map(_.toFloat)
          assert(row.length == i + 1)
          for ((x, j) <- row.zipWithIndex)
            m(i, j) = x
        }
    }

    m
  }

  def loadGRM(nSamples: Int, nVariants: Int, file: String): DenseMatrix[Float] = {
    val m: DenseMatrix[Float] = new DenseMatrix[Float](rows = nSamples, cols = nSamples)
    val rows = readFile(file, sc.hadoopConfiguration) { s =>
      val lines = Source.fromInputStream(s)
        .getLines()
        .toArray
      assert(lines.length == nSamples * (nSamples + 1) / 2)

      lines.foreach { case line =>
        val s = line.split("\t")
        val i = s(0).toInt - 1
        val j = s(1).toInt - 1
        val nNonMissing = s(2).toInt
        val x = s(3).toFloat

        assert(nNonMissing == nVariants)
        m(i, j) = x
      }
    }

    m
  }

  def readFloatLittleEndian(s: DataInputStream): Float = {
    val b0 = s.read()
    val b1 = s.read()
    val b2 = s.read()
    val b3 = s.read()
    val bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    java.lang.Float.intBitsToFloat(bits)
  }

  def loadBin(nSamples: Int, file: String): DenseMatrix[Float] = {
    val m = new DenseMatrix[Float](rows = nSamples, cols = nSamples)

    val status = hadoopFileStatus(file, sc.hadoopConfiguration)
    assert(status.getLen == 4 * nSamples * (nSamples + 1) / 2)

    readDataFile(file, sc.hadoopConfiguration) { s =>
      for (i <- 0 until nSamples)
        for (j <- 0 to i) {
          val x = readFloatLittleEndian(s)
          m(i, j) = x
        }
    }

    m
  }

  def compare(mat1: DenseMatrix[Float],
    mat2: DenseMatrix[Float]): Boolean = {

    assert(mat1.rows == mat1.cols)
    assert(mat2.rows == mat2.cols)

    if (mat1.rows != mat2.rows)
      return false

    for (i <- 0 until mat1.rows)
      for (j <- 0 to i) {
        val a = mat1(i, j)
        val b = mat2(i, j)
        if ((a - b).abs >= 1e-3) {
          println(s"$a $b")
          return false
        }
      }

    true
  }

  def runInTmp(cmd: String) {
    val pb = new java.lang.ProcessBuilder(cmd.split("\\s+"): _*)
    pb.directory(new File("/tmp"))
    pb.inheritIO()
    val p = pb.start()
    p.waitFor()
    assert(p.exitValue() == 0)
  }

  @Test def test() {

    Prop.check(forAll(VariantSampleMatrix.gen[Genotype](sc, (v: Variant) => Genotype.gen(v).filter(_.isCalled))
      // plink fails with fewer than 2 samples, no variants
      .filter(vsm => vsm.nSamples > 1 && vsm.nVariants > 0),
      Gen.oneOf("rel", "gcta-grm", "gcta-grm-bin")) {
      (vsm: VariantSampleMatrix[Genotype], format: String) =>

        var s = State(sc, sqlContext)
        s = s.copy(vds = vsm)
        s = SplitMulti.run(s, Array.empty[String])

        val sampleIds = s.vds.sampleIds
        val nSamples = s.vds.nSamples
        val nVariants = s.vds.nVariants.toInt
        assert(nVariants > 0)

        ExportPlink.run(s, Array("-o", "/tmp/plink"))

        format match {
          case "rel" =>
            runInTmp("plink --bfile /tmp/plink --make-rel")

            assert(loadIDFile("/tmp/plink.rel.id").toIndexedSeq
              == vsm.sampleIds)

            GRM.run(s, Array("--id-file", "/tmp/test.rel.id", "-f", "rel", "-o", "/tmp/test.rel"))

            assert(loadIDFile("/tmp/test.rel.id").toIndexedSeq
              == vsm.sampleIds)

            compare(loadRel(nSamples, "/tmp/plink.rel"),
              loadRel(nSamples, "/tmp/test.rel"))

          case "gcta-grm" =>
            runInTmp("plink --bfile /tmp/plink --make-grm-gz")

            assert(loadIDFile("/tmp/plink.grm.id").toIndexedSeq
              == vsm.sampleIds)

            GRM.run(s, Array("-f", "gcta-grm", "-o", "/tmp/test.grm"))

            compare(loadGRM(nSamples, nVariants, "/tmp/plink.grm.gz"),
              loadGRM(nSamples, nVariants, "/tmp/test.grm"))

          case "gcta-grm-bin" =>
            runInTmp("plink --bfile /tmp/plink --make-grm-bin")

            assert(loadIDFile("/tmp/plink.grm.id").toIndexedSeq
              == vsm.sampleIds)

            GRM.run(s, Array("-f", "gcta-grm-bin", "-o", "/tmp/test.grm.bin", "--N-file", "/tmp/test.grm.N.bin"))

            (compare(loadBin(nSamples, "/tmp/plink.grm.bin"),
              loadBin(nSamples, "/tmp/test.grm.bin"))
              && compare(loadBin(nSamples, "/tmp/plink.grm.N.bin"),
              loadBin(nSamples, "/tmp/test.grm.N.bin")))
        }
    })
  }
}
