package org.broadinstitute.k3.driver

import net.jpountz.lz4.LZ4Factory
import org.broadinstitute.k3.variant.VariantDataset

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

import org.broadinstitute.k3.methods._


object Main {
  def usage(): Unit = {
    System.err.println("usage:")
    System.err.println("")
    System.err.println("  k3 <cluster> <input> <command> [options...]")
    System.err.println("")
    System.err.println("options:")
    System.err.println("  -h, --help: print usage")
    System.err.println("")
    System.err.println("commands:")
    System.err.println("  write <output>")
    System.err.println("  nocall")
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

    //if (args.length < 3)
    //  fatal("too few arguments")

    //val master = args(0)
    //val input = args(1)
    //val command = args(2)

    val master = "local[*]"
    //val input = "/Users/Jon/sampleVCF/sample.vds"
    val input = "/Users/Jon/SampleVCF/sample.vds"
    val command = "sample"

    val conf = new SparkConf().setAppName("K3").setMaster(master)
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    // FIXME why isn't this getting picked up by from the configuration?
    conf.set("spark.executor.memory", "4g")

    val sc = new SparkContext(conf)
    // FIXME add to command line or read from CLASSPATH
    sc.addJar("/Users/cseed/k3/build/libs/k3.jar")

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val vds: VariantDataset =
      if (input.endsWith(".vds"))
        VariantDataset.read(sqlContext, input)
      else {
        if (!input.endsWith(".vcf")
          && !input.endsWith(".vcf.gz")
          && !input.endsWith(".vcfd"))
          fatal("unknown input file type")

        LoadVCF(sc, input)
      }

    println("entries: " + vds.count())

    //val output = "/Users/Jon/sampleVCF/sample.vds"
    //vds.write(sqlContext, output)

    val sampleMethods: Array[SampleMethod[Any]] =
      Array(nCalledPerSample, nNotCalledPerSample,
        nHomRefPerSample, nHetPerSample, nHomVarPerSample,
        nSNPPerSample, nIndelPerSample, nInsertionPerSample, nDeletionPerSample,
        nSingletonPerSample, nTransitionPerSample, nTransversionPerSample,
        rTiTvPerSample, rHeterozygosityPerSample, rHetHomPerSample, rDeletionInsertionPerSample)

    val variantMethods: Array[VariantMethod[Any]] =
      Array(nCalledPerVariant, nNotCalledPerVariant,
        nHomRefPerVariant, nHetPerVariant, nHomVarPerVariant,
        rHeterozygosityPerVariant, rHetHomPerVariant)

    SampleQC("/Users/Jon/sampleVCF/sampleQC.tsv", vds, sampleMethods)

    VariantQC("/Users/Jon/sampleVCF/variantQC.tsv", vds, variantMethods)


    /*
    if (command == "write") {
      //if (args.length < 4)
      //  fatal("write: too few arguments")

      //val output = args(3)
      val output = "/Users/Jon/sampleVCF/sample.vds"
      vds.write(sqlContext, output)
    } else if (command == "sample") {
      //if (args.length != 3)
      //  fatal("nocall: unexpected arguments")

      val nTransition = nTransitionPerSample(vds)
      val nTransversion = nTransversionPerSample(vds)
      val nSNPs = nSNPPerSample(vds)
      val nIndels = nIndelPerSample(vds)
      val nInsertion = nInsertionPerSample(vds)
      val nDeletion = nDeletionPerSample(vds)
      val nTiTv = nTiTvPerSample(vds)
      val tiTvRatio = rTiTvPerSample(vds)
      val nGenotypeVector = nGenotypeVectorPerSample(vds)
      val nHwe = pHwePerVariant(vds)
      val rHetHom = rHetHomPerSample(vds)
      val nHetOrHomVar = nNonRefPerVariant(vds)
      val sSingletons = sSingletonVariants(vds)

      for ((s, t) <- rHetHom)
        println(vds.sampleIds(s) + ": " + t.toString)

      for ((v, u) <- nHwe)
        println(v + ":" + u)
      println(sSingletons)
    } else if (command == "variant") {
      //if (args.length != 3)
      //  fatal("nocall: unexpected arguments")

      val variantNoCall = nHomVarPerVariant(vds)
      for ((v, nc) <- variantNoCall)
        println(v + ": " + nc)
    } else
      fatal("unknown command: " + command)
    */


  }
}
