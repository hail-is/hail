package is.hail.variant

import java.io.FileNotFoundException

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.io.plink.ExportBedBimFam
import is.hail.io.vcf.{BufferedLineIterator, ExportVCF}
import is.hail.keytable.KeyTable
import is.hail.methods._
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import is.hail.stats.ComputeRRM
import is.hail.utils._
import is.hail.variant.Variant.orderedKey
import org.apache.hadoop
import org.apache.kudu.spark.kudu._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s._

import scala.collection.mutable
import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

object VariantDataset {
  def read(hc: HailContext, dirname: String,
    metadata: VariantMetadata, parquetGenotypes: Boolean,
    skipGenotypes: Boolean = false, skipVariants: Boolean = false): VariantDataset = {

    val sqlContext = hc.sqlContext
    val sc = hc.sc
    val hConf = sc.hadoopConfiguration

    val vaSignature = metadata.vaSignature
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)

    val genotypeSignature = metadata.genotypeSignature
    val gRequiresConversion = SparkAnnotationImpex.requiresConversion(genotypeSignature)
    val isGenericGenotype = metadata.isGenericGenotype
    val isDosage = metadata.isDosage

    if (isGenericGenotype)
      fatal("Cannot read datasets with generic genotypes.")

    val parquetFile = dirname + "/rdd.parquet"

    val orderedRDD = if (skipVariants)
      OrderedRDD.empty[Locus, Variant, (Annotation, Iterable[Genotype])](sc)
    else {
      val rdd = if (skipGenotypes)
        sqlContext.readParquetSorted(parquetFile, Some(Array("variant", "annotations")))
          .map(row => (row.getVariant(0),
            (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
              Iterable.empty[Genotype])))
      else {
        val rdd = sqlContext.readParquetSorted(parquetFile)
        if (parquetGenotypes)
          rdd.map { row =>
            val v = row.getVariant(0)
            (v,
              (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
                row.getSeq[Row](2).lazyMap { rg =>
                  new RowGenotype(rg): Genotype
                }))
          } else
          rdd.map { row =>
            val v = row.getVariant(0)
            (v,
              (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
                row.getGenotypeStream(v, 2, isDosage): Iterable[Genotype]))
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

    new VariantSampleMatrix[Genotype](hc,
      if (skipGenotypes) metadata.copy(sampleIds = IndexedSeq.empty[String],
        sampleAnnotations = IndexedSeq.empty[Annotation])
      else metadata,
      orderedRDD)
  }

  def readMetadata(hConf: hadoop.conf.Configuration, dirname: String,
    requireParquetSuccess: Boolean = true): (VariantMetadata, Boolean) = {
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
           |  caught exception: ${ expandException(e, logMessage = true) }
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

    val parquetGenotypes = fields.get("parquetGenotypes") match {
      case Some(t: JBool) => t.value
      case Some(other) => fatal(
        s"""corrupt VDS: invalid metadata
           |  Expected `JBool' in field `parquetGenotypes', but got `${ other.getClass.getName }'
           |  Recreate VDS with current version of Hail.""".stripMargin)
      case _ => false
    }

    val genotypeSignature = fields.get("genotype_schema") match {
      case Some(t: JString) => Parser.parseType(t.s)
      case Some(other) => fatal(
        s"""corrupt VDS: invalid metadata
           |  Expected `JString' in field `genotype_schema', but got `${ other.getClass.getName }'
           |  Recreate VDS with current version of Hail.""".stripMargin)
      case _ => TGenotype
    }

    val isGenericGenotype = fields.get("isGenericGenotype") match {
      case Some(t: JBool) => t.value
      case Some(other) => fatal(
        s"""corrupt VDS: invalid metadata
           |  Expected `JBool' in field `isGenericGenotype', but got `${ other.getClass.getName }'
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

    (VariantMetadata(ids, annotations, globalAnnotation,
      saSignature, vaSignature, globalSignature, genotypeSignature, wasSplit, isDosage, isGenericGenotype), parquetGenotypes)
  }

  def fromKeyTable(kt: KeyTable): VariantDataset = {
    kt.keyFields.map(_.typ) match {
      case Array(TVariant) =>
      case arr => fatal("Require one key column of type Variant to produce a variant dataset, " +
        s"but found [ ${arr.mkString(", ")} ]")
    }

    val rdd = kt.keyedRDD()
      .map { case (k, v) => (k.asInstanceOf[Row].getAs[Variant](0), v) }
      .filter(_._1 != null)
      .mapValues(a => (a: Annotation, Iterable.empty[Genotype]))
      .toOrderedRDD

    val metadata = VariantMetadata(
      sampleIds = Array.empty[String],
      sa = Array.empty[Annotation],
      globalAnnotation = Annotation.empty,
      sas = TStruct.empty,
      vas = kt.valueSignature,
      globalSignature = TStruct.empty
    )

    VariantSampleMatrix[Genotype](kt.hc, metadata, rdd)
  }

  def readKudu(hc: HailContext, dirname: String, tableName: String,
    master: String): VariantDataset = {

    val (metadata, _) = readMetadata(hc.hadoopConf, dirname, requireParquetSuccess = false)
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

  def kuduRowType(vaSignature: Type): Type = TStruct("variant" -> Variant.expandedType,
    "annotations" -> vaSignature,
    "gs" -> GenotypeStream.t,
    "sample_group" -> TString)

  private def makeSchemaForKudu(vaSignature: Type): StructType =
    StructType(Array(
      StructField("variant", Variant.sparkSchema, nullable = false),
      StructField("annotations", vaSignature.schema, nullable = false),
      StructField("gs", GenotypeStream.schema, nullable = false),
      StructField("sample_group", StringType, nullable = false)
    ))
}

class VariantDatasetFunctions(private val vds: VariantSampleMatrix[Genotype]) extends AnyVal {

  private def requireSplit(methodName: String) {
    if (!vds.wasSplit)
      fatal(s"method `$methodName' requires a split dataset. Use `split_multi' or `filter_multi' first.")
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
      "s" -> (4, TString),
      "sa" -> (5, vds.saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vas3),
      "gs" -> (3, TAggregable(TGenotype, aggregationST))))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val newType = (paths, types).zipped.foldLeft(vds.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(TArray(signature), ids)
      inserterBuilder += i
      s
    }
    val finalType = if (newType.isInstanceOf[TStruct])
      paths.foldLeft(newType.asInstanceOf[TStruct]) {
        case (res, path) => res.setFieldAttributes(path, Map("Number" -> "A"))
      }
    else newType

    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vds, ec)

    vds.mapAnnotations { case (v, va, gs) =>

      val annotations = SplitMulti.split(v, va, gs,
        propagateGQ = propagateGQ,
        keepStar = true,
        isDosage = isDosage,
        insertSplitAnnots = { (va, index, wasSplit) =>
          insertSplit(insertIndex(va, index), wasSplit)
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
          inserter(va, annotations.map(_ (i)).toArray[Any]: IndexedSeq[Any])
      }

    }.copy(vaSignature = finalType)
  }

  def annotateGenotypesExpr(expr: String): GenericDataset = vds.toGDS.annotateGenotypesExpr(expr)

  def cache(): VariantDataset = persist("MEMORY_ONLY")

  def persist(storageLevel: String = "MEMORY_AND_DISK"): VariantDataset = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    val wgs = vds.withGenotypeStream()
    wgs.copy(rdd = wgs.rdd.persist(level))
  }

  def withGenotypeStream(): VariantDataset = {
    val isDosage = vds.isDosage
    vds.copy(rdd = vds.rdd.mapValuesWithKey[(Annotation, Iterable[Genotype])] { case (v, (va, gs)) =>
      (va, gs.toGenotypeStream(v, isDosage))
    }.asOrderedRDD)
  }

  def coalesce(k: Int, shuffle: Boolean = true): VariantDataset = {
    val start = if (shuffle)
      withGenotypeStream()
    else vds

    start.copy(rdd = start.rdd.coalesce(k, shuffle = shuffle)(null).asOrderedRDD)
  }

  def concordance(other: VariantDataset): (IndexedSeq[IndexedSeq[Long]], VariantDataset, VariantDataset) = {
    requireSplit("concordance")

    if (!other.wasSplit)
      fatal("method `concordance' requires both datasets to be split, but found unsplit right-hand VDS.")

    CalculateConcordance(vds, other)
  }

  def count(countGenotypes: Boolean = false): CountResult = {
    val (nVariants, nCalled) =
      if (countGenotypes) {
        val (nVar, nCalled) = vds.rdd.map { case (v, (va, gs)) =>
          (1L, gs.hardCallGenotypeIterator.countNonNegative().toLong)
        }.fold((0L, 0L)) { (comb, x) =>
          (comb._1 + x._1, comb._2 + x._2)
        }
        (nVar, Some(nCalled))
      } else
        (vds.countVariants, None)

    CountResult(vds.nSamples, nVariants, nCalled)
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

  def exportGen(path: String, precision: Int = 4) {
    requireSplit("export gen")

    def writeSampleFile() {
      //FIXME: should output all relevant sample annotations such as phenotype, gender, ...
      vds.hc.hadoopConf.writeTable(path + ".sample",
        "ID_1 ID_2 missing" :: "0 0 0" :: vds.sampleIds.map(s => s"$s $s 0").toList)
    }


    def formatDosage(d: Double): String = d.formatted(s"%.${precision}f")

    val emptyDosage = Array(0d, 0d, 0d)

    def appendRow(sb: StringBuilder, v: Variant, va: Annotation, gs: Iterable[Genotype], rsidQuery: Querier, varidQuery: Querier) {
      sb.append(v.contig)
      sb += ' '
      sb.append(Option(varidQuery(va)).getOrElse(v.toString))
      sb += ' '
      sb.append(Option(rsidQuery(va)).getOrElse("."))
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

    val localPrintRef = printRef
    val localPrintMissing = printMissing

    val filterF: Genotype => Boolean =
      g => (!g.isHomRef || localPrintRef) && (!g.isNotCalled || localPrintMissing)

    vds.exportGenotypes(path, expr, typeFile, filterF)
  }

  def exportPlink(path: String, famExpr: String = "id = s") {
    requireSplit("export plink")

    val ec = EvalContext(Map(
      "s" -> (0, TString),
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
            fatal(s"invalid type for .fam file column $i: expected $colt, got $t")
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

    val nSamples = vds.nSamples

    val plinkRDD = vds.rdd
      .mapValuesWithKey { case (v, (va, gs)) => ExportBedBimFam.makeBedRow(gs, nSamples) }
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
        val a = f().map(Option(_))
        famFns.map(_ (a)).mkString("\t")
      }

    vds.hc.hadoopConf.writeTextFile(path + ".fam")(out =>
      famRows.foreach(line => {
        out.write(line)
        out.write("\n")
      }))
  }

  /**
    *
    * @param path output path
    * @param append append file to header
    * @param exportPP export Hail PLs as a PP format field
    * @param parallel export VCF in parallel using the path argument as a directory
    */
  def exportVCF(path: String, append: Option[String] = None, exportPP: Boolean = false, parallel: Boolean = false) {
    ExportVCF(vds.toGDS, path, append, exportPP, parallel)
  }

  /**
    *
    * @param filterExpr             Filter expression involving v (variant), va (variant annotations), and aIndex (allele index)
    * @param annotationExpr         Annotation modifying expression involving v (new variant), va (old variant annotations),
    *                               and aIndices (maps from new to old indices)
    * @param filterAlteredGenotypes any call that contains a filtered allele is set to missing instead
    * @param keep                   Keep variants matching condition
    * @param subset                 subsets the PL and AD. Genotype and GQ are set based on the resulting PLs.  Downcodes by default.
    * @param maxShift               Maximum possible position change during minimum representation calculation
    */
  def filterAlleles(filterExpr: String, annotationExpr: String = "va = va", filterAlteredGenotypes: Boolean = false,
    keep: Boolean = true, subset: Boolean = true, maxShift: Int = 100, keepStar: Boolean = false): VariantDataset = {
    FilterAlleles(vds, filterExpr, annotationExpr, filterAlteredGenotypes, keep, subset, maxShift, keepStar)
  }

  /**
    *
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype), which returns a boolean value
    * @param keep keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): VariantDataset = {
    val vas = vds.vaSignature
    val sas = vds.saSignature

    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vas),
      "s" -> (2, TString),
      "sa" -> (3, sas),
      "g" -> (4, TGenotype),
      "global" -> (5, vds.globalSignature))


    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val sampleIdsBc = vds.sampleIdsBc
    val sampleAnnotationsBc = vds.sampleAnnotationsBc

    (vds.sampleIds, vds.sampleAnnotations).zipped.map((_, _))

    val noCall = Genotype()
    val localKeep = keep
    vds.mapValuesWithAll(
      (v: Variant, va: Annotation, s: String, sa: Annotation, g: Genotype) => {
        ec.setAll(v, va, s, sa, g)

        if (Filter.boxedKeepThis(f(), localKeep))
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

  /**
    *
    * @param path output path
    * @param format output format: one of rel, gcta-grm, gcta-grm-bin
    * @param idFile write ID file to this path
    * @param nFile N file path, used with gcta-grm-bin only
    */
  def grm(path: String, format: String, idFile: Option[String] = None, nFile: Option[String] = None) {
    requireSplit("GRM")
    GRM(vds, path, format, idFile, nFile)
  }

  def hardCalls(): VariantDataset = {
    vds.mapValues { g => Genotype(g.gt, g.fakeRef) }
  }

  /**
    *
    * @param computeMafExpr An expression for the minor allele frequency of the current variant, `v', given
    *                       the variant annotations `va'. If unspecified, MAF will be estimated from the dataset
    * @param bounded Allows the estimations for Z0, Z1, Z2, and PI_HAT to take on biologically-nonsense values
    *                (e.g. outside of [0,1]).
    * @param minimum Sample pairs with a PI_HAT below this value will not be included in the output. Must be in [0,1]
    * @param maximum Sample pairs with a PI_HAT above this value will not be included in the output. Must be in [0,1]
    */
  def ibd(computeMafExpr: Option[String] = None, bounded: Boolean = true,
    minimum: Option[Double] = None, maximum: Option[Double] = None): KeyTable = {
    requireSplit("IBD")

    IBD.toKeyTable(vds.hc, IBD.validateAndCall(vds, computeMafExpr, bounded, minimum, maximum))
  }

  def ibdPrune(threshold: Double, tiebreakerExpr: Option[String] = None, computeMafExpr: Option[String] = None, bounded: Boolean = true): VariantDataset = {
    requireSplit("IBD Prune")

    IBDPrune(vds, threshold, tiebreakerExpr, computeMafExpr, bounded)
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
    requireSplit("impute sex")

    val result = ImputeSexPlink(vds,
      mafThreshold,
      includePAR,
      fMaleThreshold,
      fFemaleThreshold,
      popFreqExpr)

    val signature = ImputeSexPlink.schema

    vds.annotateSamples(result, signature, "sa.imputesex")
  }

  def ldPrune(r2Threshold: Double = 0.2, windowSize: Int = 1000000, nCores: Int = 1, memoryPerCore: Int = 256): VariantDataset = {
    requireSplit("LD Prune")
    LDPrune(vds, r2Threshold, windowSize, nCores, memoryPerCore * 1024L * 1024L)
  }

  def linreg(y: String, covariates: Array[String] = Array.empty[String], root: String = "va.linreg", useDosages: Boolean = false, minAC: Int = 1, minAF: Double = 0d): VariantDataset = {
    requireSplit("linear regression")
    LinearRegression(vds, y, covariates, root, useDosages, minAC, minAF)
  }

  def linregBurden(keyName: String, variantKeys: String, singleKey: Boolean, aggExpr: String, y: String, covariates: Array[String] = Array.empty[String]): (KeyTable, KeyTable) = {
    requireSplit("linear burden regression")
    LinearRegressionBurden(vds, keyName, variantKeys, singleKey, aggExpr, y, covariates)
  }

  def linregMultiPheno(ys: Array[String], covariates: Array[String] = Array.empty[String], root: String = "va.linreg", useDosages: Boolean = false, minAC: Int = 1, minAF: Double = 0d): VariantDataset = {
    requireSplit("linear regression for multiple phenotypes")
    LinearRegressionMultiPheno(vds, ys, covariates, root, useDosages, minAC, minAF)
  }

  def lmmreg(kinshipMatrix: KinshipMatrix,
    y: String,
    covariates: Array[String] = Array.empty[String],
    useML: Boolean = false,
    rootGA: String = "global.lmmreg",
    rootVA: String = "va.lmmreg",
    runAssoc: Boolean = true,
    delta: Option[Double] = None,
    sparsityThreshold: Double = 1.0): VariantDataset = {

    requireSplit("linear mixed regression")
    LinearMixedRegression(vds, kinshipMatrix, y, covariates, useML, rootGA, rootVA,
      runAssoc, delta, sparsityThreshold)
  }

  def logreg(test: String, y: String, covariates: Array[String] = Array.empty[String], root: String = "va.logreg"): VariantDataset = {
    requireSplit("logistic regression")
    LogisticRegression(vds, test, y, covariates, root)
  }

  def logregBurden(keyName: String, variantKeys: String, singleKey: Boolean, aggExpr: String, test: String, y: String, covariates: Array[String] = Array.empty[String]): (KeyTable, KeyTable) = {
    requireSplit("linear burden regression")
    LogisticRegressionBurden(vds, keyName, variantKeys, singleKey, aggExpr, test, y, covariates)
  }

  def makeSchemaForKudu(): StructType =
    makeSchema(parquetGenotypes = false).add(StructField("sample_group", StringType, nullable = false))

  /**
    *
    * @param pathBase output root filename
    * @param famFile path to pedigree .fam file
    */
  def mendelErrors(pathBase: String, famFile: String) {
    requireSplit("mendel errors")

    val ped = Pedigree.read(famFile, vds.hc.hadoopConf, vds.sampleIds)
    val men = MendelErrors(vds, ped.completeTrios)

    men.writeMendel(pathBase + ".mendel", vds.hc.tmpDir)
    men.writeMendelL(pathBase + ".lmendel", vds.hc.tmpDir)
    men.writeMendelF(pathBase + ".fmendel")
    men.writeMendelI(pathBase + ".imendel")
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
    requireSplit("PCA")

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

  def sampleQC(root: String = "sa.qc", keepStar: Boolean = false): VariantDataset = SampleQC(vds, root, keepStar)

  def rrm(forceBlock : Boolean = false, forceGramian : Boolean = false): KinshipMatrix = {
    requireSplit("rrm")
    info(s"rrm: Computing Realized Relationship Matrix...")
    val (rrm, m) = ComputeRRM(vds, forceBlock, forceGramian)
    info(s"rrm: RRM computed using $m variants.")
    new KinshipMatrix(vds.hc, rrm, vds.sampleIds.toArray)
  }


  /**
    *
    * @param propagateGQ Propagate GQ instead of computing from PL
    * @param keepStar Do not filter * alleles
    * @param maxShift Maximum possible position change during minimum representation calculation
    */
  def splitMulti(propagateGQ: Boolean = false, keepStar: Boolean = false,
    maxShift: Int = 100): VariantDataset = {
    SplitMulti(vds, propagateGQ, keepStar, maxShift)
  }

  /**
    *
    * @param famFile path to .fam file
    * @param tdtRoot Annotation root, starting in 'va'
    */
  def tdt(famFile: String, tdtRoot: String = "va.tdt"): VariantDataset = {
    requireSplit("TDT")

    val ped = Pedigree.read(famFile, vds.hc.hadoopConf, vds.sampleIds)
    TDT(vds, ped.completeTrios,
      Parser.parseAnnotationRoot(tdtRoot, Annotation.VARIANT_HEAD))
  }

  def variantQC(root: String = "va.qc"): VariantDataset = {
    requireSplit("variant QC")
    VariantQC(vds, root)
  }

  def write(dirname: String, overwrite: Boolean = false, parquetGenotypes: Boolean = false) {
    require(dirname.endsWith(".vds"), "variant dataset write paths must end in '.vds'")

    if (overwrite)
      vds.hadoopConf.delete(dirname, recursive = true)
    else if (vds.hadoopConf.exists(dirname))
      fatal(s"file already exists at `$dirname'")

    vds.writeMetadata(dirname, parquetGenotypes = parquetGenotypes)

    val vaSignature = vds.vaSignature
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)

    val genotypeSignature = vds.genotypeSignature
    require(genotypeSignature == TGenotype, s"Expecting a genotype signature of TGenotype, but found `${genotypeSignature.toPrettyString()}'")

    vds.hadoopConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(vds.rdd.orderedPartitioner.toJSON, out)
    }

    val isDosage = vds.isDosage
    val rowRDD = vds.rdd.map { case (v, (va, gs)) =>
      Row.fromSeq(Array(v.toRow,
        if (vaRequiresConversion) SparkAnnotationImpex.exportAnnotation(va, vaSignature) else va,
        if (parquetGenotypes)
          gs.lazyMap(_.toRow).toArray[Row]: IndexedSeq[Row]
        else
          gs.toGenotypeStream(v, isDosage).toRow))
    }
    vds.hc.sqlContext.createDataFrame(rowRDD, makeSchema(parquetGenotypes = parquetGenotypes))
      .write.parquet(dirname + "/rdd.parquet")
  }

  def makeSchema(parquetGenotypes: Boolean): StructType = {
    require(!(parquetGenotypes && vds.isGenericGenotype))
    StructType(Array(
      StructField("variant", Variant.sparkSchema, nullable = false),
      StructField("annotations", vds.vaSignature.schema),
      StructField("gs",
        if (parquetGenotypes)
          ArrayType(Genotype.sparkSchema, containsNull = false)
        else
          GenotypeStream.schema,
        nullable = false)
    ))
  }

  def writeKudu(dirname: String, tableName: String,
    master: String, vcfSeqDict: String, rowsPerPartition: Int,
    sampleGroup: String, drop: Boolean = false) {
    requireSplit("write Kudu")

    vds.writeMetadata(dirname, parquetGenotypes = false)

    val vaSignature = vds.vaSignature
    val isDosage = vds.isDosage

    val rowType = VariantDataset.kuduRowType(vaSignature)
    val rowRDD = vds.rdd
      .map { case (v, (va, gs)) =>
        KuduAnnotationImpex.exportAnnotation(Annotation(
          v.toRow,
          va,
          gs.toGenotypeStream(v, isDosage).toRow,
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

    info("Written to Kudu")
  }

  def toGDS: GenericDataset = vds.mapValues(g => g: Any).copy(isGenericGenotype = true)
}
