package org.broadinstitute.k3.driver

import org.broadinstitute.k3.variant.{Variant, VariantDataset}

import scala.io.Source

import org.apache.spark.{HashPartitioner, SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._

import org.broadinstitute.k3.methods._

import scala.reflect.ClassTag

object Main {
  // FIXME
  def inject[T](r: (String, RDD[(Variant, T)]))(implicit tt: ClassTag[T]) =
    (Array[String](r._1),
      r._2.mapValues[Array[Any]](v => Array[Any](v)))
  def join[T](r1: (Array[String], RDD[(Variant, Array[Any])]), r2: (String, RDD[(Variant, T)])): (Array[String], RDD[(Variant, Array[Any])]) =
    (r1._1 :+ r2._1,
      r1._2
      .join(r2._2)
      .mapValues({ case (a, x) => a :+ x }))

  def usage(): Unit = {
    System.err.println("usage:")
    System.err.println("")
    System.err.println("  k3 <cluster> <input> <command> [options...]")
    System.err.println("")
    System.err.println("options:")
    System.err.println("  -h, --help: print usage")
    System.err.println("")
    System.err.println("commands:")
    System.err.println("  count")
    System.err.println("  repartition")
    System.err.println("  sampleqc")
    System.err.println("  variantqc")
    System.err.println("  write <output>")
  }

  def fatal(msg: String): Unit = {
    System.err.println("k3: " + msg)
    System.exit(1)
  }

  def main(args: Array[String]) {
    if (args.exists(a => a == "-h" || a == "--help")) {
      usage()
      System.exit(0)
    }

    if (args.length < 3)
      fatal("too few arguments")

    val master = args(0)
    val input = args(1)
    val command = args(2)

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

    val vds: VariantDataset =
      if (input.endsWith(".vds")) {
        val vds0 = VariantDataset.read(sqlContext, input)
        vds0
        .partitionBy(new HashPartitioner(vds0.nPartitions))
        .cache()
      } else {
        if (!input.endsWith(".vcf")
          && !input.endsWith(".vcf.gz")
          && !input.endsWith(".vcfd"))
          fatal("unknown input file type")

        LoadVCF(sc, input)
      }

    if (command == "write") {
      if (args.length < 4)
        fatal("write: too few arguments")

      val output = args(3)
      vds.write(sqlContext, output)
    } else if (command == "repartition") {
      if (args.length < 5)
        fatal("repartition: too few arguments")

      val nPartitions = args(3).toInt
      val output = args(4)

      vds
      .repartition(nPartitions)
      .write(sqlContext, output)
    } else if (command == "count") {
      if (args.length != 3)
        fatal("count: unexpected arguments")

      println("nVariants = " + vds.nVariants)
    } else if (command == "sampleqc") {
      if (args.length != 4)
        fatal("sampleqc: unexpected arguments")

      val output = args(3)

      val sampleMethods: Array[SampleMethod[Any]] =
        Array(nCalledPerSample, nNotCalledPerSample,
          nHomRefPerSample, nHetPerSample, nHomVarPerSample,
          nSNPPerSample, nIndelPerSample, nInsertionPerSample, nDeletionPerSample,
          nSingletonPerSample, nTransitionPerSample, nTransversionPerSample,
          rTiTvPerSample, rHeterozygosityPerSample, rHetHomPerSample, rDeletionInsertionPerSample)

      SampleQC(output, vds, sampleMethods)
    } else if (command == "variantqc") {
      if (args.length != 4)
        fatal("variantqc: unexpected arguments")

      val output = args(3)

      // FIXME joins bad
      val r0 = inject(nCalledPerVariant.run(vds))
      val r1 = join(r0, nNotCalledPerVariant.run(vds))
      val r2 = join(r1, nHomRefPerVariant.run(vds))
      val r3 = join(r2, nHetPerVariant.run(vds))
      val r4 = join(r3, nHomVarPerVariant.run(vds))
      val r5 = join(r4, rHeterozygosityPerVariant.run(vds))
      val r6 = join(r5, rHetHomPerVariant.run(vds))

      VariantQC(output, vds, r6)
    }
    else
      fatal("unknown command: " + command)
  }
}
