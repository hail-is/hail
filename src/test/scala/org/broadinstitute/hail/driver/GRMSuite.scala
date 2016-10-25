package org.broadinstitute.hail.driver

import java.io.DataInputStream

import breeze.linalg.DenseMatrix
import org.apache.hadoop.io.NullWritable
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils.TempDir
import org.apache.spark.WritableConverter._
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps
import scala.reflect.ClassTag
import scala.sys.process._

class GRMSuite extends SparkSuite {
  def loadIDFile(file: String): Array[String] = {
    hadoopConf.readFile(file) { s =>
      Source.fromInputStream(s)
        .getLines()
        .map { line =>
          val s = line.split("\t")
          assert(s.length == 2)
          s(1)
        }.toArray
    }
  }

  def loadHDFSMatrix[T](nSamples: Int, lines: RDD[T])(f: T => TraversableOnce[(Int, Int, Float)])(implicit ctu: ClassTag[Float]): DenseMatrix[Float] = {
    val zero: DenseMatrix[Float] = new DenseMatrix[Float](rows = nSamples, cols = nSamples)

    lines.aggregate(zero)({ case (m, line) =>
      for ((i, j, x) <- f(line)) {
        m(i, j) = x
      }
      m
    }, { case (l, r) =>
      for (i <- 0 until nSamples;
        j <- 0 until nSamples) {
        l(i, j) += r(i, j)
      }
      l
    })
  }

  def loadRelHDFS(nSamples: Int, file: String): DenseMatrix[Float] = {
    val lines = sc.textFile(file)
    assert(lines.count() == nSamples)

    val vectors = lines.map(_.split("\t").map(_.toFloat))
      .sortBy(_.length)
      .zipWithIndex()

    loadHDFSMatrix(nSamples, vectors) { case (row, i)  =>
      if (row.length != i + 1) {
        throw new AssertionError(s"row.length, ${row.length}, did not equal i + 1, ${i + 1}")
      }
      // FIXME: Possible overflow error at i.toInt
      for ((x, j) <- row.zipWithIndex) yield (i.toInt, j, x)
    }
  }

  def loadRel(nSamples: Int, file: String): DenseMatrix[Float] = {
    val m: DenseMatrix[Float] = new DenseMatrix[Float](rows = nSamples, cols = nSamples)
    val rows = hadoopConf.readFile(file) { s =>
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

  def loadGRMHDFS(nSamples: Int, nVariants: Int, file: String): DenseMatrix[Float] = {
    val lines = sc.textFile(file)
    assert(lines.count() == nSamples * (nSamples + 1) / 2)

    loadHDFSMatrix(nSamples, lines) { line =>
      val s = line.split("\t")
      val i = s(0).toInt - 1
      val j = s(1).toInt - 1
      val nNonMissing = s(2).toInt
      val x = s(3).toFloat

      if (nNonMissing != nVariants)
        throw new AssertionError(s"expected nNonMissing $nNonMissing to equal nVariants $nVariants")
      Array((i, j, x))
    }
  }

  def loadGRM(nSamples: Int, nVariants: Int, file: String): DenseMatrix[Float] = {
    val m: DenseMatrix[Float] = new DenseMatrix[Float](rows = nSamples, cols = nSamples)
    val rows = hadoopConf.readFile(file) { s =>
      val lines = Source.fromInputStream(s)
        .getLines()
        .toArray
      assert(lines.length == nSamples * (nSamples + 1) / 2)

      lines.foreach { line =>
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

    val status = hadoopConf.fileStatus(file)
    assert(status.getLen == 4 * nSamples * (nSamples + 1) / 2)

    hadoopConf.readDataFile(file) { s =>
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
          println(s"($i, $j): $a $b")
          return false
        }
      }

    true
  }

  @Test def test() {

    val bFile = tmpDir.createLocalTempFile("plink")
    val relFile = tmpDir.createTempFile("test", ".rel")
    val relIDFile = tmpDir.createTempFile("test", ".rel.id")
    val grmFile = tmpDir.createTempFile("test", ".grm")
    val grmBinFile = tmpDir.createTempFile("test", ".grm.bin")
    val grmNBinFile = tmpDir.createTempFile("test", ".grm.N.bin")

    Prop.check(forAll(
      VSMSubgen.realistic.copy(
        vGen = VariantSubgen.plinkCompatible.gen,
        tGen = VSMSubgen.realistic.tGen(_).filter(_.isCalled))
        .gen(sc)
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

        ExportPlink.run(s, Array("-o", bFile))

        format match {
          case "rel" =>
            s"plink --bfile ${ uriPath(bFile) } --make-rel --out ${ uriPath(bFile) }" !

            assert(loadIDFile(bFile + ".rel.id").toIndexedSeq
              == vsm.sampleIds)

            GRM.run(s, Array("--id-file", relIDFile, "-f", "rel", "-o", relFile))

            assert(loadIDFile(relIDFile).toIndexedSeq
              == vsm.sampleIds)

            compare(loadRel(nSamples, bFile + ".rel"),
              loadRelHDFS(nSamples, relFile))

          case "gcta-grm" =>
            s"plink --bfile ${ uriPath(bFile) } --make-grm-gz --out ${ uriPath(bFile) }" !

            assert(loadIDFile(bFile + ".grm.id").toIndexedSeq
              == vsm.sampleIds)

            GRM.run(s, Array("-f", "gcta-grm", "-o", grmFile))

            compare(loadGRM(nSamples, nVariants, bFile + ".grm.gz"),
              loadGRMHDFS(nSamples, nVariants, grmFile))

          case "gcta-grm-bin" =>
            s"plink --bfile ${ uriPath(bFile) } --make-grm-bin --out ${ uriPath(bFile) }" !

            assert(loadIDFile(bFile + ".grm.id").toIndexedSeq == vsm.sampleIds)

            GRM.run(s, Array("-f", "gcta-grm-bin", "-o", grmBinFile, "--N-file", grmNBinFile))

            (compare(loadBin(nSamples, bFile + ".grm.bin"),
              loadBin(nSamples, grmBinFile))
              && compare(loadBin(nSamples, bFile + ".grm.N.bin"),
              loadBin(nSamples, grmNBinFile)))
        }
    })
  }
}
