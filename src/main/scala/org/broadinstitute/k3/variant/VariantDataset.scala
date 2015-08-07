package org.broadinstitute.k3.variant

import java.io._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

class VariantDataset(sampleIds: Array[String],
                     rdd: RDD[(Variant, GenotypeStream)])
  extends VariantSampleMatrix[Genotype](sampleIds, rdd, (v, s, x) => x, _ => true) {

  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".vds"))

    new File(dirname).mkdir()

    val metadataOos = new ObjectOutputStream(new FileOutputStream(dirname + "/metadata.ser"))
    metadataOos.writeObject(sampleIds)

    import sqlContext.implicits._

    val df = rdd.toDF()
    df.write.parquet(dirname + "/rdd.parquet")
  }

  override def cache(): VariantDataset = new VariantDataset(sampleIds, rdd.cache())
}

object VariantDataset {
  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))

    val metadataOis = new ObjectInputStream(new FileInputStream(dirname + "/metadata.ser"))

    val sampleIdsObj = metadataOis.readObject()
    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    val sampleIds = sampleIdsObj match {
      case t: Array[String] => t
      case _ => throw new ClassCastException
    }

    import RichRow._
    new VariantDataset(sampleIds, df.rdd.map(_.toVariantGenotypeStreamTuple))
  }
}
