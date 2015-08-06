package org.broadinstitute.k3.driver

import net.jpountz.lz4.LZ4Factory
import org.broadinstitute.k3.variant.{VariantDataset}

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

import org.broadinstitute.k3.methods._

object Main {
  def main(args: Array[String]) {
    val master = if (args.length > 0)
      args(0)
    else
      "local[*]"

    val conf = new SparkConf().setAppName("K3").setMaster(master)
    conf.set("spark.sql.parquet.compression.codec", "uncompressed")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    // this is used to implicitly convert an RDD to a DataFrame.
    import sqlContext.implicits._

    // val sampleVCF = "/Users/cseed/sample.vcf.gz"
    // val sampleVCF = "/Users/cseed/swedish_scz_exomes_chr20.vcf.gz"

    // val file = "/Users/cseed/swedish_scz_exomes_chr20.vcfd"
    /*
    val file = "/Users/cseed/swedish_scz_exomes_chr20-small.vcfd"
    val vds = LoadVCF(sc, file).cache()
    println(vds.rdd.count)
    vds.write(sqlContext, "/Users/cseed/swedish_scz_exomes_chr20-small.vds")
    */

    // vs.write(sqlContext, "/Users/cseed/swedish_scz_exomes_chr20.vds")

    val vds = VariantDataset.read(sqlContext, "/Users/cseed/swedish_scz_exomes_chr20-small.vds")

    // val df = sqlContext.read.parquet("/Users/cseed/swedish_scz_exomes_chr20.parquet")
    // val v = df.rdd
    // v.cache()
    // println(v.count)

    // val df = vs.rdd.toDF()
    // df.printSchema()
    // df.write.parquet("/Users/cseed/swedish_scz_exomes_chr20.parquet")

    val sampleNoCalls = SampleNoCall(vds)

    println("SampleNoCall:")
    for (t <- vds.sampleIds.take(10).zip(sampleNoCalls))
      println(t._1 + ": " + t._2)
  }
}
