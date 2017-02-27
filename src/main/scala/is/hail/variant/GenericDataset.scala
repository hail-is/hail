package is.hail.variant

import java.io.FileNotFoundException

import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex, TGenotype, TString, TStruct}
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import is.hail.utils._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{ArrayType, StructField, StructType}
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

object GenericDataset {
  def read(hc: HailContext, dirname: String,
    skipGenotypes: Boolean = false, skipVariants: Boolean = false): GenericDataset = {

    val sqlContext = hc.sqlContext
    val sc = hc.sc
    val hConf = sc.hadoopConfiguration

    val (metadata, parquetGenotypes) = VariantDataset.readMetadata(hConf, dirname, skipGenotypes)
    val vaSignature = metadata.vaSignature
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)

    val genotypeSignature = metadata.genotypeSignature
    val gRequiresConversion = SparkAnnotationImpex.requiresConversion(genotypeSignature)
    val isGenericGenotype = metadata.isGenericGenotype
    require(isGenericGenotype, "Can only read datasets with generic genotypes.")

    val parquetFile = dirname + "/rdd.parquet"

    val orderedRDD = if (skipVariants)
      OrderedRDD.empty[Locus, Variant, (Annotation, Iterable[Annotation])](sc)
    else {
      val rdd = if (skipGenotypes)
        sqlContext.readParquetSorted(parquetFile, Some(Array("variant", "annotations")))
          .map(row => (row.getVariant(0),
            (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
              Iterable.empty[Annotation])))
      else {
        val rdd = sqlContext.readParquetSorted(parquetFile)
        rdd.map { row =>
          val v = row.getVariant(0)
          (v,
            (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
              row.getSeq[Any](2).lazyMap { g => if (gRequiresConversion) SparkAnnotationImpex.importAnnotation(g, genotypeSignature) else g }
              )
            )
        }
      }

      val partitioner: OrderedPartitioner[Locus, Variant] =
        try {
          val jv = hConf.readFile(dirname + "/partitioner.json.gz")(JsonMethods.parse(_))
          jv.fromJSON[OrderedPartitioner[Locus, Variant]]
        } catch {
          case _: FileNotFoundException =>
            fatal("missing partitioner.json.gz when loading VDS, create with HailContext.write_partitioning.")
        }

      OrderedRDD(rdd, partitioner)
    }

    new VariantSampleMatrix[Annotation](hc,
      if (skipGenotypes) metadata.copy(sampleIds = IndexedSeq.empty[String],
        sampleAnnotations = IndexedSeq.empty[Annotation])
      else metadata,
      orderedRDD)
  }
}

class GenericDatasetFunctions(private val gds: VariantSampleMatrix[Annotation]) extends AnyVal {

  def toVDS: VariantDataset = {
    if (gds.genotypeSignature != TGenotype)
      fatal(s"Cannot convert a GDS to a VDS with signature `${ gds.genotypeSignature.toPrettyString() }'")

    gds.mapValues(a => a.asInstanceOf[Genotype]).copy(isGenericGenotype = false)
  }

  def write(dirname: String, overwrite: Boolean = false): Unit = {
    require(dirname.endsWith(".vds"), "generic dataset write paths must end in '.vds'")

    if (overwrite)
      gds.hadoopConf.delete(dirname, recursive = true)
    else if (gds.hadoopConf.exists(dirname))
      fatal(s"file already exists at `$dirname'")

    writeMetadata(gds.hc.sqlContext, dirname)

    val vaSignature = gds.vaSignature
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)

    val genotypeSignature = gds.genotypeSignature
    val gRequiresConversion = SparkAnnotationImpex.requiresConversion(genotypeSignature)

    gds.hadoopConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(gds.rdd.orderedPartitioner.toJSON, out)
    }

    val rowRDD = gds.rdd.map { case (v, (va, gs)) =>
      Row.fromSeq(Array(v.toRow,
        if (vaRequiresConversion) SparkAnnotationImpex.exportAnnotation(va, vaSignature) else va,
        gs.lazyMap { g =>
          if (gRequiresConversion)
            SparkAnnotationImpex.exportAnnotation(g, genotypeSignature)
          else
            g
        }.toArray[Any]: IndexedSeq[Any]))
    }

    gds.hc.sqlContext.createDataFrame(rowRDD, makeSchema)
      .write.parquet(dirname + "/rdd.parquet")
  }

  def makeSchema: StructType = {
    StructType(Array(
      StructField("variant", Variant.schema, nullable = false),
      StructField("annotations", gds.vaSignature.schema),
      StructField("gs", ArrayType(gds.genotypeSignature.schema, containsNull = false))
    ))
  }

  def writeMetadata(sqlContext: SQLContext, dirname: String) {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"output path ending in `.vds' required, found `$dirname'")

    val hConf = gds.hc.hadoopConf
    hConf.mkDir(dirname)

    val sb = new StringBuilder

    gds.saSignature.pretty(sb, printAttrs = true, compact = true)
    val saSchemaString = sb.result()

    sb.clear()
    gds.vaSignature.pretty(sb, printAttrs = true, compact = true)
    val vaSchemaString = sb.result()

    sb.clear()
    gds.globalSignature.pretty(sb, printAttrs = true, compact = true)
    val globalSchemaString = sb.result()

    sb.clear()
    gds.genotypeSignature.pretty(sb, printAttrs = true, compact = true)
    val genotypeSchemaString = sb.result()

    val sampleInfoJson = JArray(
      gds.sampleIdsAndAnnotations
        .map { case (id, annotation) =>
          JObject(List(("id", JString(id)), ("annotation", JSONAnnotationImpex.exportAnnotation(annotation, gds.saSignature))))
        }
        .toList
    )

    val json = JObject(
      ("version", JInt(VariantSampleMatrix.fileVersion)),
      ("split", JBool(gds.wasSplit)),
      ("isDosage", JBool(gds.isDosage)),
      ("isGenericGenotype", JBool(true)),
      ("parquetGenotypes", JBool(false)),
      ("sample_annotation_schema", JString(saSchemaString)),
      ("variant_annotation_schema", JString(vaSchemaString)),
      ("global_annotation_schema", JString(globalSchemaString)),
      ("genotype_schema", JString(genotypeSchemaString)),
      ("sample_annotations", sampleInfoJson),
      ("global_annotation", JSONAnnotationImpex.exportAnnotation(gds.globalAnnotation, gds.globalSignature))
    )

    hConf.writeTextFile(dirname + "/metadata.json.gz")(Serialization.writePretty(json, _))
  }
}
