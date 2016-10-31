package org.broadinstitute.hail.keytable

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{StructField, StructType}
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.driver.Main
import org.broadinstitute.hail.expr.{BaseType, EvalContext, Parser, SparkAnnotationImpex, Type}
import org.broadinstitute.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.reflect.ClassTag


object KeyTable {
  final def fileVersion = 1

  private def readMetadata(sc: SparkContext, dirname: String, requireParquetSuccess: Boolean = true) = {
    val hConf = sc.hadoopConfiguration

    if (!hConf.exists(dirname))
      fatal(s"no key table found at `$dirname'")

    val metadataFile = dirname + "/metadata.json.gz"
    val pqtSuccess = dirname + "/rdd.parquet/_SUCCESS"

    if (!hConf.exists(pqtSuccess) && requireParquetSuccess)
      fatal(
        s"""corrupt key table: no parquet success indicator
            |  Unexpected shutdown occurred during `write'
            |  Recreate key table.""".stripMargin)

    if (!hConf.exists(metadataFile))
      fatal(
        s"""corrupt or outdated key table: invalid metadata
            |  No `metadata.json.gz' file found in key table directory
            |  Recreate key table with current version of Hail.""".stripMargin)

    val json = try {
      hConf.readFile(metadataFile)(
        in => JsonMethods.parse(in))
    } catch {
      case e: Throwable => fatal(
        s"""
           |corrupt key table: invalid metadata file.
           |  Recreate key table with current version of Hail.
           |  caught exception: ${ Main.expandException(e) }
         """.stripMargin)
    }

    val fields = json match {
      case jo: JObject => jo.obj.toMap
      case _ =>
        fatal(
          s"""corrupt key table: invalid metadata value
              |  Recreate key table with current version of Hail.""".stripMargin)
    }

    def getAndCastJSON[T <: JValue](fname: String)(implicit tct: ClassTag[T]): T =
      fields.get(fname) match {
        case Some(t: T) => t
        case Some(other) =>
          fatal(
            s"""corrupt key table: invalid metadata
                |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'
                |  Recreate key table with current version of Hail.""".stripMargin)
        case None =>
          fatal(
            s"""corrupt key table: invalid metadata
                |  Missing field `$fname'
                |  Recreate key table with current version of Hail.""".stripMargin)
      }

    val version = getAndCastJSON[JInt]("version").num

    if (version != KeyTable.fileVersion)
      fatal(
        s"""Invalid key table: old version [$version]
            |  Recreate key table with current version of Hail.
         """.stripMargin)

    val keyIdentifier = getAndCastJSON[JString]("key_identifier").s
    val keySignature = Parser.parseType(getAndCastJSON[JString]("key_schema").s)

    val valueIdentifier = getAndCastJSON[JString]("value_identifier").s
    val valueSignature = Parser.parseType(getAndCastJSON[JString]("value_schema").s)

    (keyIdentifier, valueIdentifier, keySignature, valueSignature)
  }

  def read(sqlContext: SQLContext, dirname: String, requireParquetSuccess: Boolean = true): KeyTable = {
    val sc = sqlContext.sparkContext
    val hConf = sc.hadoopConfiguration

    val (keyIdentifier, valueIdentifier, keySignature, valueSignature) = readMetadata(sc, dirname)

    val keyRequiresConversion = SparkAnnotationImpex.requiresConversion(keySignature)
    val valueRequiresConversion = SparkAnnotationImpex.requiresConversion(valueSignature)

    val parquetFile = dirname + "/rdd.parquet"

    val rdd = sqlContext.read.parquet(parquetFile)
      .map { row =>
        val k = if (keyRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(0), keySignature) else row.get(0)
        val v = if (valueRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), valueSignature) else row.get(1)
        (k, v)
      }
    KeyTable(rdd, keySignature, valueSignature, keyIdentifier, valueIdentifier)
  }

  def gen(sc: SparkContext, gen: KTSubgen): Gen[KeyTable] = gen.gen(sc)
}

case class KTSubgen(keyIdGen: Gen[String], valueIdGen: Gen[String], keySigGen: Gen[Type], valueSigGen: Gen[Type], keyAnnGen: (Type) => Gen[Annotation],
  valueAnnGen: (Type) => Gen[Annotation]) {

  def gen(sc: SparkContext): Gen[KeyTable] =
    for (size <- Gen.size;
      subsizes <- Gen.partitionSize(5).resize(size / 10);
      nPartitions <- Gen.choose(1, 10);
      keyId <- keyIdGen;
      valueId <- valueIdGen;
      keySig <- keySigGen.resize(subsizes(0));
      valueSig <- valueSigGen.resize(subsizes(1));

      (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 10) * 9);

      rows <- Gen.distinctBuildableOf[Seq, (Annotation, Annotation)](
        for (keyAnn <- keyAnnGen(keySig);
          valueAnn <- valueAnnGen(valueSig)) yield (keyAnn, valueAnn)).resize(l)

    ) yield KeyTable(sc.parallelize(rows, nPartitions), keySig, valueSig, keyId, valueId)
}

object KTSubgen {
  val random = KTSubgen(keyIdGen = Gen.identifier,
    valueIdGen = Gen.identifier,
    keySigGen = Type.genArb,
    valueSigGen = Type.genArb,
    keyAnnGen = (t: Type) => t.genValue,
    valueAnnGen = (t: Type) => t.genValue)
}

case class KeyTable(rdd: RDD[(Annotation, Annotation)], keySignature: Type, valueSignature: Type,
  keyIdentifier: String, valueIdentifier: String) {

  if (keyIdentifier == valueIdentifier)
    fatal("key identifier cannot equal value identifier in key tables")

  def same(other: KeyTable): Boolean = {
    keySignature == other.keySignature &&
      keyIdentifier == other.keyIdentifier &&
      valueSignature == other.valueSignature &&
      valueIdentifier == other.valueIdentifier &&
      rdd.groupByKey().fullOuterJoin(other.rdd.groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (None, None) => true
          case (Some(x), Some(y)) => x.toSet == y.toSet
          case _ => false
        }
      }
  }

  def schema: StructType = StructType(Array(
    StructField(keyIdentifier, keySignature.schema, nullable = false),
    StructField(valueIdentifier, valueSignature.schema, nullable = false)
  ))

  def toRowRDD: RDD[Row] = {
    val keyRequiresConversion = SparkAnnotationImpex.requiresConversion(keySignature)
    val valueRequiresConversion = SparkAnnotationImpex.requiresConversion(valueSignature)

    rdd.map { case (k, v) =>
      Row.fromSeq(Array(if (keyRequiresConversion) SparkAnnotationImpex.exportAnnotation(k, keySignature) else k,
        if (valueRequiresConversion) SparkAnnotationImpex.exportAnnotation(v, valueSignature) else v))
    }
  }

  def write(sqlContext: SQLContext, dirname: String) = {
    val sc = sqlContext.sparkContext
    val hConf = sc.hadoopConfiguration
    hConf.mkDir(dirname)

    val sb = new StringBuilder

    keySignature.pretty(sb, printAttrs = true, compact = true)
    val keySchemaString = sb.result()
    sb.clear()

    valueSignature.pretty(sb, printAttrs = true, compact = true)
    val valueSchemaString = sb.result()
    sb.clear()

    val json = JObject(
      ("version", JInt(KeyTable.fileVersion)),
      ("key_identifier", JString(keyIdentifier)),
      ("value_identifier", JString(valueIdentifier)),
      ("key_schema", JString(keySchemaString)),
      ("value_schema", JString(valueSchemaString))
    )

    hConf.writeTextFile(dirname + "/metadata.json.gz")(_.write(JsonMethods.pretty(json)))
    sqlContext.createDataFrame(toRowRDD, schema).write.parquet(dirname + "/rdd.parquet")
  }
}
