package org.broadinstitute.k3.driver

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("K3").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val sampleVCF = "/Users/cseed/sample.vcf.gz"
    // val sampleVCF = "/Users/cseed/swedish_scz_exomes_chr20.vcf.gz"
    val variantRDD = LoadVCF(sc, sampleVCF)
    
    variantRDD.take(200).foreach(println)
  }
}
