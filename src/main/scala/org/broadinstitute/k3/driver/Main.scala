package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.apache.spark.mllib.linalg.{Vector => SVector}
import org.apache.spark.sql.SQLContext
import org.broadinstitute.k3.variant.vsm.{ManagedVSM, SparkyVSM, TupleVSM}
import org.broadinstitute.k3.variant.{Genotype, VariantSampleMatrix, Variant, VariantDataset}

import scala.io.Source

import org.apache.spark.{HashPartitioner, SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._

import org.broadinstitute.k3.methods._

import scala.reflect.ClassTag

object Main {
  def usage() {
    System.err.println("usage:")
    System.err.println("")
    System.err.println("  k3 <input> <command> [options...]")
    System.err.println("")
    System.err.println("<input> can be a .vcf, .vcf.gz, .vcfd or .vds file.")
    System.err.println("")
    System.err.println("options:")
    System.err.println("  -h, --help: print usage")
    System.err.println("  --master <master>: use Spark cluster master <master>")
    System.err.println("  --vsmtype <type>: use VariantSampleMatrix implementation <type>")
    System.err.println("")
    System.err.println("commands:")
    System.err.println("  count")
    System.err.println("  gqbydp <output .tsv>")
    System.err.println("  pca <output .tsv>")
    System.err.println("  repartition <nPartitions> <output .vds>")
    System.err.println("  sampleqc <output .tsv>")
    System.err.println("  variantqc <output .tsv>")
    System.err.println("  write <output .vds>")
  }

  def fatal(msg: String): Nothing = {
    System.err.println("k3: " + msg)
    System.exit(1)
    throw new AssertionError
  }

  def main(args: Array[String]) {
    if (args.exists(a => a == "-h" || a == "--help")) {
      usage()
      System.exit(0)
    }

    if (args.length < 2)
      fatal("too few arguments")

    var vsmtype = "sparky"
    var master = "local[*]"
    var command: String = null
    var input: String = null
    var filter: String = null

    var argi = 0
    var mainArgsDone = false
    do {
      if (argi == args.length) {
        fatal("too few arguments")
        System.exit(0)
      }

      val arg = args(argi)
      argi += 1
      arg match {
        case "--master" =>
          if (argi == args.length)
            fatal("--master: argument expected")
          master = args(argi)
          argi += 1
        case "--vsmtype" =>
          if (argi == args.length)
            fatal("--vsmtype: argument expected")
          vsmtype = args(argi)
          argi += 1
        case "--filter" =>
          if (argi == args.length)
            fatal("--filter: argument expected")
          filter = args(argi)
          argi += 1
        case s: String if s(0) == '-' =>
          fatal("unknown option `" + s + "'")
        case s: String =>
          if (input == null)
            input = s
          else if (command == null) {
            command = s
            mainArgsDone = true
          }
      }
    } while (!mainArgsDone)
    assert(command != null && input != null)

    println("k3: master = " + master)
    println("k3: vsmtype = " + vsmtype)
    println("k3: command = " + command)
    println("k3: input = " + input)

    val conf = new SparkConf().setAppName("K3").setMaster(master)
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    // FIXME why isn't this getting picked up by from the configuration?
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "2g")

    val sc = new SparkContext(conf)

    val jar = getClass().getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
    sc.addJar(jar)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val rawVDS: VariantSampleMatrix[Genotype] =
      if (input.endsWith(".vds")) {
        VariantSampleMatrix.read(sqlContext, vsmtype, input)
        .cache()
      } else {
        if (!input.endsWith(".vcf")
          && !input.endsWith(".vcf.gz")
          && !input.endsWith(".vcfd"))
          fatal("unknown input file type")

        LoadVCF(sc, vsmtype, input)
      }

    val vds: VariantDataset = if (filter != null) {
      filter match {
        case f if f.endsWith(".interval_list") =>
          rawVDS.filterVariants(LoadIntervalList(filter))
        case "isSNP" =>
          rawVDS.filterVariants(v => v.isSNP)
        case "isIndel" =>
          rawVDS.filterVariants(v => v.isIndel)
        case _ =>
          fatal("unknown filter option")
      }
    } else
      rawVDS

    if (command == "write") {
      if (argi != args.length - 1)
        fatal("write: wrong number of arguments")
      val output = args(argi)
      argi += 1

      vds.write(sqlContext, output)
    } else if (command == "repartition") {
      if (argi != args.length - 2)
        fatal("repartition: too few arguments")

      val nPartitions = args(argi).toInt
      argi += 1
      val output = args(argi)
      argi += 1

      vds
      .repartition(nPartitions)
      .write(sqlContext, output)
    } else if (command == "count") {
      if (argi != args.length)
        fatal("count: unexpected arguments")

      println("count = " + vds.count)
    } else if (command == "sampleqc") {
      if (argi != args.length - 1)
        fatal("sampleqc: unexpected arguments")

      val output = args(argi)
      argi += 1

      val sampleMethods: Array[SampleMethod[Any]] =
        Array(nCalledPerSample, nNotCalledPerSample,
          nHomRefPerSample, nHetPerSample, nHomVarPerSample,
          nSNPPerSample, nIndelPerSample, nInsertionPerSample, nDeletionPerSample,
          nSingletonPerSample, nTransitionPerSample, nTransversionPerSample,
          rTiTvPerSample, rHeterozygosityPerSample, rHetHomPerSample, rDeletionInsertionPerSample)

      SampleQC(output, vds, sampleMethods)
    } else if (command == "variantqc") {
      if (argi != args.length - 1)
        fatal("variantqc: unexpected arguments")

      val output = args(argi)
      argi += 1

      // FIXME joins bad
      def inject[T](r: (String, RDD[(Variant, T)]))(implicit tt: ClassTag[T]) =
        (Array[String](r._1),
          r._2.mapValues[Array[Any]](v => Array[Any](v)))
      def join[T](r1: (Array[String], RDD[(Variant, Array[Any])]), r2: (String, RDD[(Variant, T)])): (Array[String], RDD[(Variant, Array[Any])]) =
        (r1._1 :+ r2._1,
          r1._2
          .join(r2._2)
          .mapValues({ case (a, x) => a :+ x }))

      val r0 = inject(nCalledPerVariant.run(vds))
      val r1 = join(r0, nNotCalledPerVariant.run(vds))
      val r2 = join(r1, nHomRefPerVariant.run(vds))
      val r3 = join(r2, nHetPerVariant.run(vds))
      val r4 = join(r3, nHomVarPerVariant.run(vds))
      val r5 = join(r4, rHeterozygosityPerVariant.run(vds))
      val r6 = join(r5, rHetHomPerVariant.run(vds))

      VariantQC(output, vds, r6)
    } else if (command == "gqbydp") {
      if (argi != args.length - 1)
        fatal("gqbydp: unexpected arguments")

      val output = args(argi)
      argi += 1

      val fw = new FileWriter(new File(output))

      val nBins = GQByDPBins.nBins
      val binStep = GQByDPBins.binStep
      val firstBinLow = GQByDPBins.firstBinLow

      fw.write("sample")
      for (b <- 0 until nBins) {
        fw.write("\t" + GQByDPBins.binLow(b) + "-" + GQByDPBins.binHigh(b))
      }
      fw.write("\n")

      val gqbydp = GQByDPBins(vds)
      for (i <- vds.sampleIds.indices) {
        fw.write(vds.sampleIds(i))
        for (b <- 0 until GQByDPBins.nBins) {
          gqbydp.get((i, b)) match {
            case Some(percentGQ) => fw.write("\t" + percentGQ)
            case None => fw.write("\t-")
          }
        }
        fw.write("\n")
      }

      fw.close()
    } else if (command ==  "pca") {
      if (argi != args.length - 1)
        fatal("pca: unexpected arguments")

      val output = args(argi)
      argi += 1

      val k = 10
      val samplePCs = new SamplePCA(k)(vds)

      val fw = new FileWriter(new File(output))
      fw.write("sample")
      for (i <- 0 until k)
        fw.write("\t" + "PC" + i)
      fw.write("\n")

      for (i <- 0 until vds.nSamples) {
        fw.write(vds.sampleIds(i))
        for (j <- 0 until k)
          fw.write("\t" + samplePCs(i)(j))
        fw.write("\n")
      }

      fw.close()
    } else
      fatal("unknown command: " + command)
  }
}
