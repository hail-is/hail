package is.hail.variant

import java.io.FileNotFoundException

import is.hail.annotations.{Annotation, _}
import is.hail.driver.HailContext
import is.hail.expr.{EvalContext, JSONAnnotationImpex, Parser, SparkAnnotationImpex, TString, TStruct, Type, _}
import is.hail.io._
import is.hail.io.annotators.{BedAnnotator, IntervalListAnnotator}
import is.hail.io.plink.{ExportBedBimFam, FamFileConfig, PlinkLoader}
import is.hail.io.vcf.{BufferedLineIterator, ExportVCF}
import is.hail.methods._
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import is.hail.utils._
import is.hail.variant.Variant.orderedKey
import org.apache.hadoop
import org.apache.kudu.spark.kudu.{KuduContext, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{JArray, JBool, JInt, JObject, JString, JValue, _}

import scala.collection.mutable
import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

object VariantDataset {
  private def readMetadata(hConf: hadoop.conf.Configuration, dirname: String,
    requireParquetSuccess: Boolean = true): VariantMetadata = {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"input path ending in `.vds' required, found `$dirname'")

    if (!hConf.exists(dirname))
      fatal(s"no VDS found at `$dirname'")

    val metadataFile = dirname + "/metadata.json.gz"
    val pqtSuccess = dirname + "/rdd.parquet/_SUCCESS"

    if (!hConf.exists(pqtSuccess) && requireParquetSuccess)
      fatal(
        s"""corrupt VDS: no parquet success indicator
           |  Unexpected shutdown occurred during `write'
           |  Recreate VDS.""".stripMargin)

    if (!hConf.exists(metadataFile))
      fatal(
        s"""corrupt or outdated VDS: invalid metadata
           |  No `metadata.json.gz' file found in VDS directory
           |  Recreate VDS with current version of Hail.""".stripMargin)

    val json = try {
      hConf.readFile(metadataFile)(
        in => JsonMethods.parse(in))
    } catch {
      case e: Throwable => fatal(
        s"""
           |corrupt VDS: invalid metadata file.
           |  Recreate VDS with current version of Hail.
           |  caught exception: ${ expandException(e) }
         """.stripMargin)
    }

    val fields = json match {
      case jo: JObject => jo.obj.toMap
      case _ =>
        fatal(
          s"""corrupt VDS: invalid metadata value
             |  Recreate VDS with current version of Hail.""".stripMargin)
    }

    def getAndCastJSON[T <: JValue](fname: String)(implicit tct: ClassTag[T]): T =
      fields.get(fname) match {
        case Some(t: T) => t
        case Some(other) =>
          fatal(
            s"""corrupt VDS: invalid metadata
               |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'
               |  Recreate VDS with current version of Hail.""".stripMargin)
        case None =>
          fatal(
            s"""corrupt VDS: invalid metadata
               |  Missing field `$fname'
               |  Recreate VDS with current version of Hail.""".stripMargin)
      }

    val version = getAndCastJSON[JInt]("version").num

    if (version != VariantSampleMatrix.fileVersion)
      fatal(
        s"""Invalid VDS: old version [$version]
           |  Recreate VDS with current version of Hail.
         """.stripMargin)

    val wasSplit = getAndCastJSON[JBool]("split").value
    val isDosage = fields.get("isDosage") match {
      case Some(t: JBool) => t.value
      case Some(other) => fatal(
        s"""corrupt VDS: invalid metadata
           |  Expected `JBool' in field `isDosage', but got `${ other.getClass.getName }'
           |  Recreate VDS with current version of Hail.""".stripMargin)
      case _ => false
    }

    val saSignature = Parser.parseType(getAndCastJSON[JString]("sample_annotation_schema").s)
    val vaSignature = Parser.parseType(getAndCastJSON[JString]("variant_annotation_schema").s)
    val globalSignature = Parser.parseType(getAndCastJSON[JString]("global_annotation_schema").s)

    val sampleInfoSchema = TStruct(("id", TString), ("annotation", saSignature))
    val sampleInfo = getAndCastJSON[JArray]("sample_annotations")
      .arr
      .map {
        case JObject(List(("id", JString(id)), ("annotation", jv: JValue))) =>
          (id, JSONAnnotationImpex.importAnnotation(jv, saSignature, "sample_annotations"))
        case other => fatal(
          s"""corrupt VDS: invalid metadata
             |  Invalid sample annotation metadata
             |  Recreate VDS with current version of Hail.""".stripMargin)
      }
      .toArray

    val globalAnnotation = JSONAnnotationImpex.importAnnotation(getAndCastJSON[JValue]("global_annotation"),
      globalSignature, "global")

    val ids = sampleInfo.map(_._1)
    val annotations = sampleInfo.map(_._2)

    VariantMetadata(ids, annotations, globalAnnotation,
      saSignature, vaSignature, globalSignature, wasSplit, isDosage)
  }

  def read(hc: HailContext, dirname: String,
    skipGenotypes: Boolean = false, skipVariants: Boolean = false): VariantDataset = {

    val sqlContext = hc.sqlContext
    val sc = hc.sc
    val hConf = sc.hadoopConfiguration

    val metadata = readMetadata(hConf, dirname, skipGenotypes)
    val vaSignature = metadata.vaSignature

    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)
    val isDosage = metadata.isDosage

    val parquetFile = dirname + "/rdd.parquet"

    val orderedRDD = if (skipVariants)
      OrderedRDD.empty[Locus, Variant, (Annotation, Iterable[Genotype])](sc)
    else {
      val rdd = if (skipGenotypes)
        sqlContext.readParquetSorted(parquetFile, Some(Array("variant", "annotations")))
          .map(row => (row.getVariant(0),
            (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
              Iterable.empty[Genotype])))
      else
        sqlContext.readParquetSorted(parquetFile)
          .map { row =>
            val v = row.getVariant(0)
            (v,
              (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
                row.getGenotypeStream(v, 2, isDosage): Iterable[Genotype]))
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

    new VariantSampleMatrix[Genotype](hc,
      if (skipGenotypes) metadata.copy(sampleIds = IndexedSeq.empty[String],
        sampleAnnotations = IndexedSeq.empty[Annotation])
      else metadata,
      orderedRDD)
  }

  def kuduRowType(vaSignature: Type): Type = TStruct("variant" -> Variant.t,
    "annotations" -> vaSignature,
    "gs" -> GenotypeStream.t,
    "sample_group" -> TString)

  def readKudu(hc: HailContext, dirname: String, tableName: String,
    master: String): VariantDataset = {

    val metadata = readMetadata(hc.hadoopConf, dirname, requireParquetSuccess = false)
    val vaSignature = metadata.vaSignature
    val isDosage = metadata.isDosage

    val df = hc.sqlContext.read.options(
      Map("kudu.table" -> tableName, "kudu.master" -> master)).kudu

    val rowType = kuduRowType(vaSignature)
    val schema: StructType = KuduAnnotationImpex.exportType(rowType).asInstanceOf[StructType]

    // Kudu key fields are always first, so we have to reorder the fields we get back
    // to be in the column order for the flattened schema *before* we unflatten
    val indices: Array[Int] = schema.fields.zipWithIndex.map { case (field, rowIdx) =>
      df.schema.fieldIndex(field.name)
    }

    val rdd: RDD[(Variant, (Annotation, Iterable[Genotype]))] = df.rdd.map { row =>
      val importedRow = KuduAnnotationImpex.importAnnotation(
        KuduAnnotationImpex.reorder(row, indices), rowType).asInstanceOf[Row]
      val v = importedRow.getVariant(0)
      (v,
        (importedRow.get(1),
          importedRow.getGenotypeStream(v, 2, metadata.isDosage)))
    }.spanByKey().map(kv => {
      // combine variant rows with different sample groups (no shuffle)
      val variant = kv._1
      val annotations = kv._2.head._1
      // just use first annotation
      val genotypes = kv._2.flatMap(_._2) // combine genotype streams
      (variant, (annotations, genotypes))
    })
    new VariantSampleMatrix[Genotype](hc, metadata, rdd.toOrderedRDD)
  }

  private def makeSchemaForKudu(vaSignature: Type): StructType =
    StructType(Array(
      StructField("variant", Variant.schema, nullable = false),
      StructField("annotations", vaSignature.schema, nullable = false),
      StructField("gs", GenotypeStream.schema, nullable = false),
      StructField("sample_group", StringType, nullable = false)
    ))
}

class VariantDatasetFunctions(private val vds: VariantSampleMatrix[Genotype]) extends AnyVal {

  private def rdd = vds.rdd

  def makeSchema(): StructType =
    StructType(Array(
      StructField("variant", Variant.schema, nullable = false),
      StructField("annotations", vds.vaSignature.schema),
      StructField("gs", GenotypeStream.schema, nullable = false)
    ))

  def makeSchemaForKudu(): StructType =
    makeSchema().add(StructField("sample_group", StringType, nullable = false))

  def coalesce(k: Int, shuffle: Boolean = true): VariantDataset = {
    val start = if (shuffle)
      withGenotypeStream()
    else vds
    vds.copy(rdd = vds.rdd)
    start.copy(rdd = rdd.coalesce(k, shuffle = shuffle)(null).asOrderedRDD)
  }

  private def writeMetadata(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"output path ending in `.vds' required, found `$dirname'")

    val hConf = vds.hc.hadoopConf
    hConf.mkDir(dirname)

    val sb = new StringBuilder

    vds.saSignature.pretty(sb, printAttrs = true, compact = true)
    val saSchemaString = sb.result()

    sb.clear()
    vds.vaSignature.pretty(sb, printAttrs = true, compact = true)
    val vaSchemaString = sb.result()

    sb.clear()
    vds.globalSignature.pretty(sb, printAttrs = true, compact = true)
    val globalSchemaString = sb.result()

    val sampleInfoSchema = TStruct(("id", TString), ("annotation", vds.saSignature))
    val sampleInfoJson = JArray(
      vds.sampleIdsAndAnnotations
        .map { case (id, annotation) =>
          JObject(List(("id", JString(id)), ("annotation", JSONAnnotationImpex.exportAnnotation(annotation, vds.saSignature))))
        }
        .toList
    )

    val json = JObject(
      ("version", JInt(VariantSampleMatrix.fileVersion)),
      ("split", JBool(vds.wasSplit)),
      ("isDosage", JBool(vds.isDosage)),
      ("sample_annotation_schema", JString(saSchemaString)),
      ("variant_annotation_schema", JString(vaSchemaString)),
      ("global_annotation_schema", JString(globalSchemaString)),
      ("sample_annotations", sampleInfoJson),
      ("global_annotation", JSONAnnotationImpex.exportAnnotation(vds.globalAnnotation, vds.globalSignature))
    )

    hConf.writeTextFile(dirname + "/metadata.json.gz")(Serialization.writePretty(json, _))
  }

  def write(dirname: String, overwrite: Boolean = false, compress: Boolean = true) {
    require(dirname.endsWith(".vds"), "variant dataset write paths must end in '.vds'")

    if (overwrite)
      vds.hadoopConf.delete(dirname, recursive = true)
    else if (vds.hadoopConf.exists(dirname))
      fatal(s"file already exists at `$dirname'")

    writeMetadata(vds.hc.sqlContext, dirname, compress)

    val vaSignature = vds.vaSignature
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)

    val ordered = vds.rdd.asOrderedRDD

    vds.hadoopConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(ordered.orderedPartitioner.toJSON, out)
    }

    val isDosage = vds.isDosage
    val rowRDD = ordered.map { case (v, (va, gs)) =>
      Row.fromSeq(Array(v.toRow,
        if (vaRequiresConversion) SparkAnnotationImpex.exportAnnotation(va, vaSignature) else va,
        gs.toGenotypeStream(v, isDosage, compress).toRow))
    }
    vds.hc.sqlContext.createDataFrame(rowRDD, makeSchema())
      .write.parquet(dirname + "/rdd.parquet")
    // .saveAsParquetFile(dirname + "/rdd.parquet")
  }

  def writeKudu(dirname: String, tableName: String,
    master: String, vcfSeqDict: String, rowsPerPartition: Int,
    sampleGroup: String, compress: Boolean = true, drop: Boolean = false) {

    writeMetadata(vds.hc.sqlContext, dirname, compress)

    val vaSignature = vds.vaSignature
    val isDosage = vds.isDosage

    val rowType = VariantDataset.kuduRowType(vaSignature)
    val rowRDD = vds.rdd
      .map { case (v, (va, gs)) =>
        KuduAnnotationImpex.exportAnnotation(Annotation(
          v.toRow,
          va,
          gs.toGenotypeStream(v, isDosage, compress).toRow,
          sampleGroup), rowType).asInstanceOf[Row]
      }

    val schema: StructType = KuduAnnotationImpex.exportType(rowType).asInstanceOf[StructType]
    println(s"schema = $schema")
    val df = vds.hc.sqlContext.createDataFrame(rowRDD, schema)

    val kuduContext = new KuduContext(master)
    if (drop) {
      KuduUtils.dropTable(master, tableName)
      Thread.sleep(10 * 1000) // wait to avoid overwhelming Kudu service queue
    }
    if (!KuduUtils.tableExists(master, tableName)) {
      val hConf = vds.hc.sqlContext.sparkContext.hadoopConfiguration
      val headerLines = hConf.readFile(vcfSeqDict) { s =>
        Source.fromInputStream(s)
          .getLines()
          .takeWhile { line => line(0) == '#' }
          .toArray
      }
      val codec = new htsjdk.variant.vcf.VCFCodec()
      val seqDict = codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
        .getHeaderValue
        .asInstanceOf[htsjdk.variant.vcf.VCFHeader]
        .getSequenceDictionary

      val keys = Seq("variant__contig", "variant__start", "variant__ref",
        "variant__altAlleles_0__alt", "sample_group")
      kuduContext.createTable(tableName, schema, keys,
        KuduUtils.createTableOptions(schema, keys, seqDict, rowsPerPartition))
    }
    df.write
      .options(Map("kudu.master" -> master, "kudu.table" -> tableName))
      .mode("append")
      // FIXME inlined since .kudu wouldn't work for some reason
      .format("org.apache.kudu.spark.kudu").save

    println("Written to Kudu")
  }

  def eraseSplit(): VariantDataset = {
    if (vds.wasSplit) {
      val (newSignatures1, f1) = vds.deleteVA("wasSplit")
      val vds1 = vds.copy(vaSignature = newSignatures1)
      val (newSignatures2, f2) = vds1.deleteVA("aIndex")
      vds1.copy(wasSplit = false,
        vaSignature = newSignatures2,
        rdd = vds1.rdd.mapValuesWithKey { case (v, (va, gs)) =>
          (f2(f1(va)), gs.lazyMap(g => g.copy(fakeRef = false)))
        }.asOrderedRDD)
    } else
      vds
  }

  def withGenotypeStream(compress: Boolean = true): VariantDataset = {
    val isDosage = vds.isDosage
    vds.copy(rdd = vds.rdd.mapValuesWithKey[(Annotation, Iterable[Genotype])] { case (v, (va, gs)) =>
      (va, gs.toGenotypeStream(v, isDosage, compress = compress))
    }.asOrderedRDD)
  }

  def aggregateIntervals(intervalList: String, expr: String, out: String) {

    val vas = vds.vaSignature
    val sas = vds.saSignature
    val localGlobalAnnotation = vds.globalAnnotation

    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "interval" -> (1, TInterval),
      "v" -> (2, TVariant),
      "va" -> (3, vds.vaSignature))
    val symTab = Map(
      "global" -> (0, vds.globalSignature),
      "interval" -> (1, TInterval),
      "variants" -> (2, TAggregable(TVariant, aggregationST)))

    val ec = EvalContext(symTab)
    ec.set(1, vds.globalAnnotation)

    val (names, _, f) = Parser.parseExportExprs(expr, ec)

    if (names.isEmpty)
      fatal("this module requires one or more named expr arguments")

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[(Interval[Locus], Variant, Annotation)](ec, { case (ec, (i, v, va)) =>
      ec.setAll(localGlobalAnnotation, i, v, va)
    })

    val iList = IntervalListAnnotator.read(intervalList, vds.hc.hadoopConf)
    val iListBc = vds.sparkContext.broadcast(iList)

    val results = vds.variantsAndAnnotations.flatMap { case (v, va) =>
      iListBc.value.query(v.locus).map { i => (i, (i, v, va)) }
    }
      .aggregateByKey(zVals)(seqOp, combOp)
      .collectAsMap()

    vds.hc.hadoopConf.writeTextFile(out) { out =>
      val sb = new StringBuilder
      sb.append("Contig")
      sb += '\t'
      sb.append("Start")
      sb += '\t'
      sb.append("End")
      names.foreach { col =>
        sb += '\t'
        sb.append(col)
      }
      sb += '\n'

      iList.toIterator
        .foreachBetween { interval =>

          sb.append(interval.start.contig)
          sb += '\t'
          sb.append(interval.start.position)
          sb += '\t'
          sb.append(interval.end.position)
          val res = results.getOrElse(interval, zVals)
          resultOp(res)

          ec.setAll(localGlobalAnnotation, interval)
          f().foreach { field =>
            sb += '\t'
            sb.append(field)
          }
        }(sb += '\n')

      out.write(sb.result())
    }
  }

  def annotateAllelesExpr(expr: String, propagateGQ: Boolean = false): VariantDataset = {
    val isDosage = vds.isDosage

    val (vas2, insertIndex) = vds.vaSignature.insert(TInt, "aIndex")
    val (vas3, insertSplit) = vas2.insert(TBoolean, "wasSplit")
    val localGlobalAnnotation = vds.globalAnnotation

    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vas3),
      "g" -> (3, TGenotype),
      "s" -> (4, TSample),
      "sa" -> (5, vds.saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vas3),
      "gs" -> (3, TAggregable(TGenotype, aggregationST))))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(TArray(signature), ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vds, ec)

    vds.mapAnnotations { case (v, va, gs) =>

      val annotations = SplitMulti.split(v, va, gs,
        propagateGQ = propagateGQ,
        compress = true,
        keepStar = true,
        isDosage = isDosage,
        insertSplitAnnots = { (va, index, wasSplit) =>
          insertSplit(insertIndex(va, Some(index)), Some(wasSplit))
        },
        f = _ => true)
        .map({
          case (v, (va, gs)) =>
            ec.setAll(localGlobalAnnotation, v, va)
            aggregateOption.foreach(f => f(v, va, gs))
            f()
        }).toArray

      inserters.zipWithIndex.foldLeft(va) {
        case (va, (inserter, i)) =>
          inserter(va, Some(annotations.map(_ (i).getOrElse(Annotation.empty)).toArray[Any]: IndexedSeq[Any]))
      }

    }.copy(vaSignature = finalType)
  }

  def annotateGlobalExpr(expr: String): VariantDataset = {
    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature)))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Option(Annotation.GLOBAL_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalType = (paths, types).zipped.foldLeft(vds.globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    ec.set(0, vds.globalAnnotation)
    val ga = inserters
      .zip(f())
      .foldLeft(vds.globalAnnotation) { case (a, (ins, res)) =>
        ins(a, res)
      }

    vds.copy(globalAnnotation = ga,
      globalSignature = finalType)
  }

  def annotateGlobalList(path: String, root: String, asSet: Boolean = false): VariantDataset = {
    val textList = vds.hc.hadoopConf.readFile(path) { in =>
      Source.fromInputStream(in)
        .getLines()
        .toArray
    }

    val (sig, toInsert) =
      if (asSet)
        (TSet(TString), textList.toSet)
      else
        (TArray(TString), textList: IndexedSeq[String])

    val rootPath = Parser.parseAnnotationRoot(root, "global")

    val (newGlobalSig, inserter) = vds.insertGlobal(sig, rootPath)

    vds.copy(
      globalAnnotation = inserter(vds.globalAnnotation, Some(toInsert)),
      globalSignature = newGlobalSig)
  }

  def annotateGlobalTable(path: String, root: String,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.GLOBAL_HEAD)

    val (struct, rdd) = TextTableReader.read(vds.sparkContext)(Array(path), config)
    val arrayType = TArray(struct)

    val (finalType, inserter) = vds.insertGlobal(arrayType, annotationPath)

    val table = rdd
      .map(_.value)
      .collect(): IndexedSeq[Annotation]

    vds.copy(
      globalAnnotation = inserter(vds.globalAnnotation, Some(table)),
      globalSignature = finalType)
  }

  def annotateSamplesExpr(expr: String): VariantDataset = {
    val ec = Aggregators.sampleEC(vds)

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.SAMPLE_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.saSignature) { case (sas, (ids, signature)) =>
      val (s, i) = sas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, ec)

    ec.set(0, vds.globalAnnotation)
    val newAnnotations = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.set(1, s)
      ec.set(2, sa)
      f().zip(inserters)
        .foldLeft(sa) { case (sa, (v, inserter)) =>
          inserter(sa, v)
        }
    }

    vds.copy(
      sampleAnnotations = newAnnotations,
      saSignature = finalType
    )
  }

  def annotateSamplesFam(path: String, root: String = "sa.fam",
    config: FamFileConfig = FamFileConfig()): VariantDataset = {
    if (!path.endsWith(".fam"))
      fatal("input file must end in .fam")

    val (info, signature) = PlinkLoader.parseFam(path, config, vds.hc.hadoopConf)

    val duplicateIds = info.map(_._1).duplicates().toArray
    if (duplicateIds.nonEmpty) {
      val n = duplicateIds.length
      fatal(
        s"""found $n duplicate sample ${ plural(n, "id") }:
           |  @1""".stripMargin, duplicateIds)
    }

    vds.annotateSamples(info.toMap, signature, root)
  }

  def annotateSamplesList(path: String, root: String): VariantDataset = {

    val samplesInList = vds.hc.hadoopConf.readLines(path) { lines =>
      if (lines.isEmpty)
        warn(s"Empty annotation file given: $path")

      lines.map(_.value).toSet
    }

    val sampleAnnotations = vds.sampleIds.map { s => (s, samplesInList.contains(s)) }.toMap
    vds.annotateSamples(sampleAnnotations, TBoolean, root)
  }

  def annotateSamplesTable(path: String, sampleExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (struct, rdd) = TextTableReader.read(vds.sparkContext)(Array(path), config)

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "sa" -> (0, vds.saSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vds.saSignature, ec, Annotation.SAMPLE_HEAD)
      } else
        vds.insertSA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.SAMPLE_HEAD))

    val sampleQuery = struct.parseInStructScope[String](sampleExpr)

    val map = rdd
      .flatMap {
        _.map { a =>
          sampleQuery(a).map(s => (s, a))
        }.value
      }
      .collect()
      .toMap

    val vdsKeys = vds.sampleIds.toSet
    val tableKeys = map.keySet
    val onlyVds = vdsKeys -- tableKeys
    val onlyTable = tableKeys -- vdsKeys
    if (onlyVds.nonEmpty) {
      warn(s"There were ${onlyVds.size} samples present in the VDS but not in the table.")
    }
    if (onlyTable.nonEmpty) {
      warn(s"There were ${onlyTable.size} samples present in the table but not in the VDS.")
    }


    vds.annotateSamples(map.get _, finalType, inserter)
  }

  def annotateSamplesVDS(other: VariantDataset,
    root: Option[String] = None,
    code: Option[String] = None): VariantDataset = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "sa" -> (0, vds.saSignature),
          "vds" -> (1, other.saSignature)))
        Annotation.buildInserter(annotationExpr, vds.saSignature, ec, Annotation.SAMPLE_HEAD)
      } else
        vds.insertSA(other.saSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.SAMPLE_HEAD))

    val m = other.sampleIdsAndAnnotations.toMap
    vds
      .annotateSamples(m.get _, finalType, inserter)
  }

  def annotateVariantsBED(path: String, root: String, all: Boolean = false): VariantDataset = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    BedAnnotator(path, vds.hc.hadoopConf) match {
      case (is, None) =>
        vds.annotateIntervals(is, annotationPath)

      case (is, Some((t, m))) =>
        vds.annotateIntervals(is, t, m, all = all, annotationPath)
    }
  }

  def annotateVariantsExpr(expr: String): VariantDataset = {
    val localGlobalAnnotation = vds.globalAnnotation

    val ec = Aggregators.variantEC(vds)
    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vds, ec)

    vds.mapAnnotations { case (v, va, gs) =>
      ec.setAll(localGlobalAnnotation, v, va)

      aggregateOption.foreach(f => f(v, va, gs))
      f().zip(inserters)
        .foldLeft(va) { case (va, (v, inserter)) =>
          inserter(va, v)
        }
    }.copy(vaSignature = finalType)
  }

  def annotateVariantsIntervals(path: String, root: String, all: Boolean = false): VariantDataset = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    IntervalListAnnotator(path, vds.hc.hadoopConf) match {
      case (is, Some((m, t))) =>
        vds.annotateIntervals(is, m, t, all = all, annotationPath)

      case (is, None) =>
        vds.annotateIntervals(is, annotationPath)
    }
  }

  def annotateVariantsLoci(path: String, locusExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    annotateVariantsLociAll(List(path), locusExpr, root, code, config)
  }

  def annotateVariantsLociAll(paths: Seq[String], locusExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    val files = vds.hc.hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, rdd) = TextTableReader.read(vds.sparkContext)(files, config, vds.nPartitions)

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vds.vaSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
      } else vds.insertVA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    val locusQuery = struct.parseInStructScope[Locus](locusExpr)


    import is.hail.variant.LocusImplicits.orderedKey
    val lociRDD = rdd.flatMap {
      _.map { a =>
        locusQuery(a).map(l => (l, a))
      }.value
    }.toOrderedRDD(vds.rdd.orderedPartitioner.mapMonotonic)

    vds.annotateLoci(lociRDD, finalType, inserter)
  }

  def annotateVariantsTable(path: String, variantExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    annotateVariantsTables(List(path), variantExpr, root, code, config)
  }

  def annotateVariantsTables(paths: Seq[String], variantExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    val files = vds.hc.hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, rdd) = TextTableReader.read(vds.sparkContext)(files, config, vds.nPartitions)

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vds.vaSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
      } else vds.insertVA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    val variantQuery = struct.parseInStructScope[Variant](variantExpr)

    val keyedRDD = rdd.flatMap {
      _.map { a =>
        variantQuery(a).map(v => (v, a))
      }.value
    }.toOrderedRDD(vds.rdd.orderedPartitioner)

    vds.annotateVariants(keyedRDD, finalType, inserter)
  }

  def annotateVariantsVDS(other: VariantDataset,
    root: Option[String] = None, code: Option[String] = None): VariantDataset = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vds.vaSignature),
          "vds" -> (1, other.vaSignature)))
        Annotation.buildInserter(annotationExpr, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
      } else vds.insertVA(other.vaSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    vds.annotateVariants(other.variantsAndAnnotations, finalType, inserter)
  }

  def cache(): VariantDataset = persist("MEMORY_ONLY")

  def concordance(other: VariantDataset): (IndexedSeq[IndexedSeq[Long]], VariantDataset, VariantDataset) = {
    require(vds.wasSplit && other.wasSplit, "method `concordance' requires both left and right datasets to be split.")

    CalculateConcordance(vds, other)
  }

  def count(countGenotypes: Boolean = false): CountResult = {
    val (nVariants, nCalled) =
      if (countGenotypes) {
        val (nVar, nCalled) = vds.rdd.map { case (v, (va, gs)) =>
          (1L, gs.count(_.isCalled).toLong)
        }.fold((0L, 0L)) { (comb, x) =>
          (comb._1 + x._1, comb._2 + x._2)
        }
        (nVar, Some(nCalled))
      } else
        (vds.countVariants, None)

    CountResult(vds.nSamples, nVariants, nCalled)
  }

  def deduplicate(): VariantDataset = {
    DuplicateReport.initialize()

    val acc = DuplicateReport.accumulator
    vds.copy(rdd = vds.rdd.mapPartitions({ it =>
      new SortedDistinctPairIterator(it, (v: Variant) => acc += v)
    }, preservesPartitioning = true).asOrderedRDD)
  }

  def downsampleVariants(keep: Long): VariantDataset = {
    vds.sampleVariants(keep.toDouble / vds.countVariants())
  }

  def exportGen(path: String) {
    require(vds.wasSplit, "method `exportGen' requires a split dataset")

    def writeSampleFile() {
      //FIXME: should output all relevant sample annotations such as phenotype, gender, ...
      vds.hc.hadoopConf.writeTable(path + ".sample",
        "ID_1 ID_2 missing" :: "0 0 0" :: vds.sampleIds.map(s => s"$s $s 0").toList)
    }


    def formatDosage(d: Double): String = d.formatted("%.4f")

    val emptyDosage = Array(0d, 0d, 0d)

    def appendRow(sb: StringBuilder, v: Variant, va: Annotation, gs: Iterable[Genotype], rsidQuery: Querier, varidQuery: Querier) {
      sb.append(v.contig)
      sb += ' '
      sb.append(varidQuery(va).getOrElse(v.toString))
      sb += ' '
      sb.append(rsidQuery(va).getOrElse("."))
      sb += ' '
      sb.append(v.start)
      sb += ' '
      sb.append(v.ref)
      sb += ' '
      sb.append(v.alt)

      for (gt <- gs) {
        val dosages = gt.dosage.getOrElse(emptyDosage)
        sb += ' '
        sb.append(formatDosage(dosages(0)))
        sb += ' '
        sb.append(formatDosage(dosages(1)))
        sb += ' '
        sb.append(formatDosage(dosages(2)))
      }
    }

    def writeGenFile() {
      val varidSignature = vds.vaSignature.getOption("varid")
      val varidQuery: Querier = varidSignature match {
        case Some(_) => val (t, q) = vds.queryVA("va.varid")
          t match {
            case TString => q
            case _ => a => None
          }
        case None => a => None
      }

      val rsidSignature = vds.vaSignature.getOption("rsid")
      val rsidQuery: Querier = rsidSignature match {
        case Some(_) => val (t, q) = vds.queryVA("va.rsid")
          t match {
            case TString => q
            case _ => a => None
          }
        case None => a => None
      }

      val isDosage = vds.isDosage

      vds.rdd.mapPartitions { it: Iterator[(Variant, (Annotation, Iterable[Genotype]))] =>
        val sb = new StringBuilder
        it.map { case (v, (va, gs)) =>
          sb.clear()
          appendRow(sb, v, va, gs, rsidQuery, varidQuery)
          sb.result()
        }
      }.writeTable(path + ".gen", vds.hc.tmpDir, None)
    }

    writeSampleFile()
    writeGenFile()
  }

  def exportGenotypes(path: String, expr: String, typeFile: Boolean,
    printRef: Boolean = false, printMissing: Boolean = false) {
    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "g" -> (4, TGenotype),
      "global" -> (5, vds.globalSignature))

    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)
    val (names, ts, f) = Parser.parseExportExprs(expr, ec)

    val hadoopConf = vds.hc.hadoopConf
    if (typeFile) {
      hadoopConf.delete(path + ".types", recursive = false)
      val typeInfo = names
        .getOrElse(ts.indices.map(i => s"_$i").toArray)
        .zip(ts)
      exportTypes(path + ".types", hadoopConf, typeInfo)
    }

    hadoopConf.delete(path, recursive = true)

    val sampleIdsBc = vds.sparkContext.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = vds.sparkContext.broadcast(vds.sampleAnnotations)

    val localPrintRef = printRef
    val localPrintMissing = printMissing

    val filterF: Genotype => Boolean =
      g => (!g.isHomRef || localPrintRef) && (!g.isNotCalled || localPrintMissing)

    val lines = vds.mapPartitionsWithAll { it =>
      val sb = new StringBuilder()
      it
        .filter { case (v, va, s, sa, g) => filterF(g) }
        .map { case (v, va, s, sa, g) =>
          ec.setAll(v, va, s, sa, g)
          sb.clear()

          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
    }.writeTable(path, vds.hc.tmpDir, names.map(_.mkString("\t")))
  }

  def exportPlink(path: String, famExpr: String = "id = s.id") {
    require(vds.wasSplit, "method `exportPlink' requires a split dataset")

    val ec = EvalContext(Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature),
      "global" -> (2, vds.globalSignature)))

    ec.set(2, vds.globalAnnotation)

    type Formatter = (Option[Any]) => String

    val formatID: Formatter = _.map(_.asInstanceOf[String]).getOrElse("0")
    val formatIsFemale: Formatter = _.map { a =>
      if (a.asInstanceOf[Boolean])
        "2"
      else
        "1"
    }.getOrElse("0")
    val formatIsCase: Formatter = _.map { a =>
      if (a.asInstanceOf[Boolean])
        "2"
      else
        "1"
    }.getOrElse("-9")
    val formatQPheno: Formatter = a => a.map(_.toString).getOrElse("-9")

    val famColumns: Map[String, (Type, Int, Formatter)] = Map(
      "famID" -> (TString, 0, formatID),
      "id" -> (TString, 1, formatID),
      "patID" -> (TString, 2, formatID),
      "matID" -> (TString, 3, formatID),
      "isFemale" -> (TBoolean, 4, formatIsFemale),
      "qPheno" -> (TDouble, 5, formatQPheno),
      "isCase" -> (TBoolean, 5, formatIsCase))

    val (names, types, f) = Parser.parseNamedExprs(famExpr, ec)

    val famFns: Array[(Array[Option[Any]]) => String] = Array(
      _ => "0", _ => "0", _ => "0", _ => "0", _ => "-9", _ => "-9")

    (names.zipWithIndex, types).zipped.foreach { case ((name, i), t) =>
      famColumns.get(name) match {
        case Some((colt, j, formatter)) =>
          if (colt != t)
            fatal("invalid type for .fam file column $h: expected $colt, got $t")
          famFns(j) = (a: Array[Option[Any]]) => formatter(a(i))

        case None =>
          fatal(s"no .fam file column $name")
      }
    }

    val spaceRegex = """\s+""".r
    val badSampleIds = vds.sampleIds.filter(id => spaceRegex.findFirstIn(id).isDefined)
    if (badSampleIds.nonEmpty) {
      fatal(
        s"""Found ${ badSampleIds.length } sample IDs with whitespace
           |  Please run `renamesamples' to fix this problem before exporting to plink format
           |  Bad sample IDs: @1 """.stripMargin, badSampleIds)
    }

    val bedHeader = Array[Byte](108, 27, 1)

    val plinkRDD = vds.rdd
      .mapValuesWithKey { case (v, (va, gs)) => ExportBedBimFam.makeBedRow(gs) }
      .persist(StorageLevel.MEMORY_AND_DISK)

    plinkRDD.map { case (v, bed) => bed }
      .saveFromByteArrays(path + ".bed", vds.hc.tmpDir, header = Some(bedHeader))

    plinkRDD.map { case (v, bed) => ExportBedBimFam.makeBimRow(v) }
      .writeTable(path + ".bim", vds.hc.tmpDir)

    plinkRDD.unpersist()

    val famRows = vds
      .sampleIdsAndAnnotations
      .map { case (s, sa) =>
        ec.setAll(s, sa)
        val a = f()
        famFns.map(_ (a)).mkString("\t")
      }

    vds.hc.hadoopConf.writeTextFile(path + ".fam")(out =>
      famRows.foreach(line => {
        out.write(line)
        out.write("\n")
      }))
  }

  def exportSamples(path: String, expr: String, typeFile: Boolean = false) {
    val localGlobalAnnotation = vds.globalAnnotation

    val ec = Aggregators.sampleEC(vds)

    val (names, types, f) = Parser.parseExportExprs(expr, ec)
    val hadoopConf = vds.hc.hadoopConf
    if (typeFile) {
      hadoopConf.delete(path + ".types", recursive = false)
      val typeInfo = names
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(path + ".types", hadoopConf, typeInfo)
    }

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, ec)

    hadoopConf.delete(path, recursive = true)

    val sb = new StringBuilder()
    val lines = for ((s, sa) <- vds.sampleIdsAndAnnotations) yield {
      sampleAggregationOption.foreach(f => f.apply(s))
      sb.clear()
      ec.setAll(localGlobalAnnotation, s, sa)
      f().foreachBetween(x => sb.append(x))(sb += '\t')
      sb.result()
    }

    hadoopConf.writeTable(path, lines, names.map(_.mkString("\t")))
  }

  def exportVariants(path: String, expr: String, typeFile: Boolean = false) {
    val vas = vds.vaSignature
    val hConf = vds.hc.hadoopConf

    val localGlobalAnnotations = vds.globalAnnotation
    val ec = Aggregators.variantEC(vds)

    val (names, types, f) = Parser.parseExportExprs(expr, ec)

    val hadoopConf = vds.hc.hadoopConf
    if (typeFile) {
      hadoopConf.delete(path + ".types", recursive = false)
      val typeInfo = names
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(path + ".types", hadoopConf, typeInfo)
    }

    val variantAggregations = Aggregators.buildVariantAggregations(vds, ec)

    hadoopConf.delete(path, recursive = true)

    vds.rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (v, (va, gs)) =>
          variantAggregations.foreach { f => f(v, va, gs) }
          ec.setAll(localGlobalAnnotations, v, va)
          sb.clear()
          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
      }.writeTable(path, vds.hc.tmpDir, names.map(_.mkString("\t")))
  }

  /**
    *
    * @param address Cassandra contact point to connect to
    * @param keySpace Cassandra keyspace
    * @param table Cassandra table
    * @param genotypeExpr comma-separated list of fields/computations to be exported
    * @param variantExpr comma-separated list of fields/computations to be exported
    * @param drop drop and re-create Cassandra table before exporting
    * @param exportRef export HomRef calls
    * @param exportMissing export missing genotypes
    * @param blockSize size of exported batch
    */
  def exportVariantsCassandra(address: String, genotypeExpr: String, keySpace: String,
    table: String, variantExpr: String, drop: Boolean = false, exportRef: Boolean = false,
    exportMissing: Boolean = false, blockSize: Int = 100) {

    CassandraConnector.exportVariants(vds, address, keySpace, table, genotypeExpr,
      variantExpr, drop, exportRef, exportMissing, blockSize)
  }

  /**
    *
    * @param variantExpr comma-separated list of fields/computations to be exported
    * @param genotypeExpr comma-separated list of fields/computations to be exported
    * @param collection SolrCloud collection
    * @param url Solr instance (URL) to connect to
    * @param zkHost Zookeeper host string to connect to
    * @param exportMissing export missing genotypes
    * @param exportRef export HomRef calls
    * @param drop delete and re-create solr collection before exporting
    * @param numShards number of shards to split the collection into
    * @param blockSize Variants per SolrClient.add
    */
  def exportVariantsSolr(variantExpr: String,
    genotypeExpr: String,
    collection: String = null,
    url: String = null,
    zkHost: String = null,
    exportMissing: Boolean = false,
    exportRef: Boolean = false,
    drop: Boolean = false,
    numShards: Int = 1,
    blockSize: Int = 100) {

    SolrConnector.exportVariants(vds, variantExpr, genotypeExpr, collection, url, zkHost, exportMissing,
      exportRef, drop, numShards, blockSize)
  }

  /**
    *
    * @param path output path
    * @param append append file to header
    * @param exportPP export Hail PLs as a PP format field
    * @param parallel export VCF in parallel using the path argument as a directory
    */
  def exportVCF(path: String, append: Option[String] = None, exportPP: Boolean = false, parallel: Boolean = false) {
    ExportVCF(vds, path, append, exportPP, parallel)
  }

  /**
    *
    * @param filterExpr Filter expression involving v (variant), va (variant annotations), and aIndex (allele index)
    * @param annotationExpr Annotation modifying expression involving v (new variant), va (old variant annotations),
    *                       and aIndices (maps from new to old indices)
    * @param filterAlteredGenotypes any call that contains a filtered allele is set to missing instead
    * @param remove Remove variants matching condition
    * @param downcode downcodes the PL and AD. Genotype and GQ are set based on the resulting PLs
    * @param subset subsets the PL and AD. Genotype and GQ are set based on the resulting PLs
    * @param maxShift Maximum possible position change during minimum representation calculation
    */
  def filterAlleles(filterExpr: String, annotationExpr: String = "va = va", filterAlteredGenotypes: Boolean = false,
    remove: Boolean = false, downcode: Boolean = false, subset: Boolean = false, maxShift: Int = 100): VariantDataset = {
    FilterAlleles(vds, filterExpr, annotationExpr, filterAlteredGenotypes, remove, downcode, subset, maxShift)
  }

  /**
    *
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype), which returns a boolean value
    * @param remove remove genotypes where filterExpr evaluates to true, rather than keep them
    */
  def filterGenotypes(filterExpr: String, remove: Boolean = false): VariantDataset = {
    val vas = vds.vaSignature
    val sas = vds.saSignature

    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas),
      "s" -> (2, TSample),
      "sa" -> (3, sas),
      "g" -> (4, TGenotype),
      "global" -> (5, vds.globalSignature))


    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)
    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](filterExpr, ec)

    val sampleIdsBc = vds.sampleIdsBc
    val sampleAnnotationsBc = vds.sampleAnnotationsBc

    (vds.sampleIds, vds.sampleAnnotations).zipped.map((_, _))

    val noCall = Genotype()
    val keep = !remove
    vds.mapValuesWithAll(
      (v: Variant, va: Annotation, s: String, sa: Annotation, g: Genotype) => {
        ec.setAll(v, va, s, sa, g)

        if (Filter.keepThis(f(), keep))
          g
        else
          noCall
      })
  }

  /**
    * Remove multiallelic variants from this dataset.
    *
    * Useful for running methods that require biallelic variants without calling the more expensive split_multi step.
    */
  def filterMulti(): VariantDataset = {
    if (vds.wasSplit) {
      warn("called redundant `filtermulti' on an already split or multiallelic-filtered VDS")
      vds
    } else {
      vds.filterVariants {
        case (v, va, gs) => v.isBiallelic
      }.copy(wasSplit = true)
    }
  }

  /**
    * Filter samples using the Hail expression language.
    *
    * @param filterExpr Filter expression involving `s' (sample) and `sa' (sample annotations)
    * @param remove remove samples where filterExpr evaluates to true, rather than keeping only these
    */
  def filterSamplesExpr(filterExpr: String, remove: Boolean = false): VariantDataset = {
    val localGlobalAnnotation = vds.globalAnnotation

    val keep = !remove
    val sas = vds.saSignature

    val ec = Aggregators.sampleEC(vds)

    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](filterExpr, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, ec)

    val sampleIds = vds.sampleIds
    val p = (s: String, sa: Annotation) => {
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotation, s, sa)
      Filter.keepThis(f(), keep)
    }

    vds.filterSamples(p)
  }

  /**
    * Filter samples using a text file containing sample IDs
    * @param path path to sample list file
    * @param remove remove listed samples rather than keeping them
    */
  def filterSamplesList(path: String, remove: Boolean = false): VariantDataset = {
    val samples = vds.hc.hadoopConf.readFile(path) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.isEmpty)
        .toSet
    }
    val p = (s: String, sa: Annotation) => Filter.keepThis(samples.contains(s), !remove)

    vds.filterSamples(p)
  }

  /**
    * Filter variants using the Hail expression language.
    * @param filterExpr filter expression
    * @param remove remove variants where filterExpr evaluates to true, rather than keeping only these
    * @return
    */
  def filterVariantsExpr(filterExpr: String, remove: Boolean = false): VariantDataset = {
    val localGlobalAnnotation = vds.globalAnnotation
    val ec = Aggregators.variantEC(vds)

    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](filterExpr, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations(vds, ec)

    val p = (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
      aggregatorOption.foreach(f => f(v, va, gs))

      ec.setAll(localGlobalAnnotation, v, va)
      Filter.keepThis(f(), !remove)
    }

    vds.filterVariants(p)
  }

  /**
    *
    * @param path output path
    * @param format output format: one of rel, gcta-grm, gcta-grm-bin
    * @param idFile write ID file to this path
    * @param nFile N file path, used with gcta-grm-bin only
    */
  def grm(path: String, format: String, idFile: Option[String] = None, nFile: Option[String] = None) {
    GRM(vds, path, format, idFile, nFile)
  }

  def gqByDP(path: String) {
    val nBins = GQByDPBins.nBins
    val binStep = GQByDPBins.binStep
    val firstBinLow = GQByDPBins.firstBinLow
    val gqbydp = GQByDPBins(vds)

    vds.hadoopConf.writeTextFile(path) { s =>
      s.write("sample")
      for (b <- 0 until nBins)
        s.write("\t" + GQByDPBins.binLow(b) + "-" + GQByDPBins.binHigh(b))

      s.write("\n")

      for (sample <- vds.sampleIds) {
        s.write(sample)
        for (b <- 0 until GQByDPBins.nBins) {
          gqbydp.get((sample, b)) match {
            case Some(percentGQ) => s.write("\t" + percentGQ)
            case None => s.write("\tNA")
          }
        }
        s.write("\n")
      }
    }
  }

  def hardCalls(): VariantDataset = {
    vds.mapValues { g => Genotype(g.gt, g.fakeRef) }
  }

  /**
    *
    * @param path Output path for the IBD matrix
    * @param computeMafExpr An expression for the minor allele frequency of the current variant, `v', given
    *                       the variant annotations `va'. If unspecified, MAF will be estimated from the dataset
    * @param bounded Allows the estimations for Z0, Z1, Z2, and PI_HAT to take on biologically-nonsense values
    *                (e.g. outside of [0,1]).
    * @param parallelWrite This option writes the IBD table as a directory of sharded files. Without this option,
    *                      the output is coalesced to a single file
    * @param minimum Sample pairs with a PI_HAT below this value will not be included in the output. Must be in [0,1]
    * @param maximum Sample pairs with a PI_HAT above this value will not be included in the output. Must be in [0,1]
    */
  def ibd(path: String, computeMafExpr: Option[String] = None, bounded: Boolean = true, parallelWrite: Boolean = false,
    minimum: Option[Double] = None, maximum: Option[Double] = None) {

    minimum.foreach(min => optionCheckInRangeInclusive(0.0, 1.0)("minimum", min))
    maximum.foreach(max => optionCheckInRangeInclusive(0.0, 1.0)("maximum", max))

    minimum.liftedZip(maximum).foreach { case (min, max) =>
      if (min <= max) {
        fatal(s"minimum must be less than or equal to maximum: $min, $max")
      }
    }

    val computeMaf = computeMafExpr.map(IBD.generateComputeMaf(vds.vaSignature, _))

    IBD(vds, computeMaf, bounded, minimum, maximum)
      .map { case ((i, j), ibd) =>
        s"$i\t$j\t${ ibd.ibd.Z0 }\t${ ibd.ibd.Z1 }\t${ ibd.ibd.Z2 }\t${ ibd.ibd.PI_HAT }"
      }
      .writeTable(path, vds.hc.tmpDir, Some("SAMPLE_ID_1\tSAMPLE_ID_2\tZ0\tZ1\tZ2\tPI_HAT"), parallelWrite)
  }

  /**
    *
    * @param mafThreshold Minimum minor allele frequency threshold
    * @param includePAR Include pseudoautosomal regions
    * @param fFemaleThreshold Samples are called females if F < femaleThreshold
    * @param fMaleThreshold Samples are called males if F > maleThreshold
    * @param popFreqExpr Use an annotation expression for estimate of MAF rather than computing from the data
    */
  def imputeSex(mafThreshold: Double = 0.0, includePAR: Boolean = false, fFemaleThreshold: Double = 0.2,
    fMaleThreshold: Double = 0.8, popFreqExpr: Option[String] = None): VariantDataset = {

    val result = ImputeSexPlink(vds,
      mafThreshold,
      includePAR,
      fMaleThreshold,
      fFemaleThreshold,
      popFreqExpr)

    val signature = ImputeSexPlink.schema

    vds.annotateSamples(result, signature, "sa.imputesex")
  }

  /**
    *
    * @param right right-hand dataset with which to join
    */
  def join(right: VariantDataset): VariantDataset = {
    if (vds.wasSplit != right.wasSplit) {
      warn(
        s"""cannot join split and unsplit datasets
           |  left was split: ${ vds.wasSplit }
           |  light was split: ${ right.wasSplit }""".stripMargin)
    }

    if (vds.saSignature != right.saSignature) {
      fatal(
        s"""cannot join datasets with different sample schemata
           |  left sample schema: @1
           |  right sample schema: @2""".stripMargin,
        vds.saSignature.toPrettyString(compact = true, printAttrs = true),
        right.saSignature.toPrettyString(compact = true, printAttrs = true))
    }

    val newSampleIds = vds.sampleIds ++ right.sampleIds
    val duplicates = newSampleIds.duplicates()
    if (duplicates.nonEmpty)
      fatal("duplicate sample IDs: @1", duplicates)

    val joined = vds.rdd.orderedInnerJoinDistinct(right.rdd)
      .mapValues { case ((lva, lgs), (rva, rgs)) =>
        (lva, lgs ++ rgs)
      }.asOrderedRDD

    vds.copy(
      sampleIds = newSampleIds,
      sampleAnnotations = vds.sampleAnnotations ++ right.sampleAnnotations,
      rdd = joined)
  }

  def linreg(ySA: String, covSA: Array[String], root: String, minAC: Int, minAF: Double): VariantDataset = {
    LinearRegression(vds, ySA, covSA, root, minAC, minAF)
  }

  def logreg(test: String, ySA: String, covSA: Array[String], root: String): VariantDataset = {
    LogisticRegression(vds, test, ySA, covSA, root)
  }

  /**
    *
    * @param pathBase output root filename
    * @param famFile path to pedigree .fam file
    */
  def mendelErrors(pathBase: String, famFile: String) {

    val ped = Pedigree.read(famFile, vds.hc.hadoopConf, vds.sampleIds)
    val men = MendelErrors(vds, ped.completeTrios)

    men.writeMendel(pathBase + ".mendel", vds.hc.tmpDir)
    men.writeMendelL(pathBase + ".lmendel", vds.hc.tmpDir)
    men.writeMendelF(pathBase + ".fmendel")
    men.writeMendelI(pathBase + ".imendel")
  }

  def persist(storageLevel: String = "MEMORY_AND_DISK"): VariantDataset = {
    vds.rdd.persist()
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    vds.withGenotypeStream().copy(rdd = vds.rdd.persist(level))
  }

  /**
    *
    * @param scoresRoot Sample annotation path for scores (period-delimited path starting in 'sa')
    * @param k Number of principal components
    * @param loadingsRoot Variant annotation path for site loadings (period-delimited path starting in 'va')
    * @param eigenRoot Global annotation path for eigenvalues (period-delimited path starting in 'global'
    * @param asArrays Store score and loading results as arrays, rather than structs
    */
  def pca(scoresRoot: String, k: Int = 10, loadingsRoot: Option[String] = None, eigenRoot: Option[String] = None,
    asArrays: Boolean = false): VariantDataset = {
    if (k < 1)
      fatal(
        s"""requested invalid number of components: $k
           |  Expect componenents >= 1""".stripMargin)

    info(s"Running PCA with $k components...")

    val pcSchema = SamplePCA.pcSchema(asArrays, k)

    val (scores, loadings, eigenvalues) =
      SamplePCA(vds, k, loadingsRoot.isDefined, eigenRoot.isDefined, asArrays)

    var ret = vds.annotateSamples(scores, pcSchema, scoresRoot)

    loadings.foreach { rdd =>
      ret = ret.annotateVariants(rdd.orderedRepartitionBy(vds.rdd.orderedPartitioner), pcSchema, loadingsRoot.get)
    }

    eigenvalues.foreach { eig =>
      ret = ret.annotateGlobal(eig, pcSchema, eigenRoot.get)
    }
    ret
  }

  /**
    *
    * @param path ID mapping file
    */
  def renameSamples(path: String): VariantDataset = {
    val m = vds.hc.hadoopConf.readFile(path) { s =>
      Source.fromInputStream(s)
        .getLines()
        .map {
          _.split("\t") match {
            case Array(old, news) => (old, news)
            case _ =>
              fatal("Invalid input. Use two tab-separated columns.")
          }
        }.toMap
    }

    val newSamples = mutable.Set.empty[String]
    val newSampleIds = vds.sampleIds
      .map { s =>
        val news = m.getOrElse(s, s)
        if (newSamples.contains(news))
          fatal(s"duplicate sample ID `$news' after rename")
        newSamples += news
        news
      }
    vds.copy(sampleIds = newSampleIds)
  }

  def sampleQC(): VariantDataset = SampleQC(vds)

  /**
    *
    * @param propagateGQ Propagate GQ instead of computing from PL
    * @param compress Don't compress genotype streams
    * @param keepStar Do not filter * alleles
    * @param maxShift Maximum possible position change during minimum representation calculation
    */
  def splitMulti(propagateGQ: Boolean = false, compress: Boolean = true, keepStar: Boolean = false,
    maxShift: Int = 100): VariantDataset = {
    SplitMulti(vds, propagateGQ, compress, keepStar, maxShift)
  }

  /**
    *
    * @param famFile path to .fam file
    * @param tdtRoot Annotation root, starting in 'va'
    */
  def tdt(famFile: String, tdtRoot: String = "va.tdt"): VariantDataset = {
    val ped = Pedigree.read(famFile, vds.hc.hadoopConf, vds.sampleIds)
    TDT(vds, ped.completeTrios,
      Parser.parseAnnotationRoot(tdtRoot, Annotation.VARIANT_HEAD))
  }

  def typecheck() {
    var foundError = false
    if (!vds.globalSignature.typeCheck(vds.globalAnnotation)) {
      warn(
        s"""found violation in global annotation
           |Schema: ${ vds.globalSignature.toPrettyString() }
           |
            |Annotation: ${ Annotation.printAnnotation(vds.globalAnnotation) }""".stripMargin)
    }

    vds.sampleIdsAndAnnotations.find { case (_, sa) => !vds.saSignature.typeCheck(sa) }
      .foreach { case (s, sa) =>
        foundError = true
        warn(
          s"""found violation in sample annotations for sample $s
             |Schema: ${ vds.saSignature.toPrettyString() }
             |
              |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val vaSignature = vds.vaSignature
    vds.variantsAndAnnotations.find { case (_, va) => !vaSignature.typeCheck(va) }
      .foreach { case (v, va) =>
        foundError = true
        warn(
          s"""found violation in variant annotations for variant $v
             |Schema: ${ vaSignature.toPrettyString() }
             |
              |Annotation: ${ Annotation.printAnnotation(va) }""".stripMargin)
      }

    if (foundError)
      fatal("found one or more type check errors")
  }

  def variantQC(): VariantDataset = VariantQC(vds)

  /**
    *
    * @param config VEP configuration file
    * @param root Variant annotation path to store VEP output
    * @param csq Annotates with the VCF CSQ field as a string, rather than the full nested struct schema
    * @param force Force VEP annotation from scratch
    * @param blockSize Variants per VEP invocation
    */
  def vep(config: String, root: String = "va.vep", csq: Boolean = false, force: Boolean = false,
    blockSize: Int = 1000): VariantDataset = {
    VEP.annotate(vds, config, root, csq, force, blockSize)
  }
}