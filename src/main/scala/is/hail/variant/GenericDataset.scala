package is.hail.variant

import java.io.FileNotFoundException

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{EvalContext, JSONAnnotationImpex, Parser, SparkAnnotationImpex, TGenotype, TString, TStruct, TVariant, Type}
import is.hail.io.vcf.ExportVCF
import is.hail.methods.Filter
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import is.hail.utils._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{ArrayType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.mutable

class GenericDatasetFunctions(private val gds: GenericDataset) extends AnyVal {
  type M = GenericMatrixT

  def annotateGenotypesExpr(expr: String): GenericDataset = {
    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, gds.vaSignature),
      "s" -> (2, TString),
      "sa" -> (3, gds.saSignature),
      "g" -> (4, gds.genotypeSignature),
      "global" -> (5, gds.globalSignature))


    val ec = EvalContext(symTab)
    ec.set(5, gds.globalAnnotation)

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.GENOTYPE_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(gds.genotypeSignature) { case (gsig, (ids, signature)) =>
      val (s, i) = gsig.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    info(s"""Modified the genotype schema with AnnotateGenotypesExpr.
             |  Original: ${gds.genotypeSignature.toPrettyString(compact = true)}
             |  New: ${finalType.toPrettyString(compact = true)}""".stripMargin)

    gds.mapValuesWithAll[M](
      (v: Variant, va: Annotation, s: String, sa: Annotation, g: Annotation) => {
        ec.setAll(v, va, s, sa, g)
        f().zip(inserters)
          .foldLeft(g) { case (ga, (a, inserter)) =>
            inserter(ga, a)
          }
      }).copy[M](genotypeSignature = finalType)
  }

  def cache(): GenericDataset = persist("MEMORY_ONLY")

  def coalesce(k: Int, shuffle: Boolean = true): GenericDataset =
    gds.copy[M](rdd = gds.rdd.coalesce(k, shuffle = shuffle)(null).toOrderedRDD)

  def exportGenotypes(path: String, expr: String, typeFile: Boolean, printMissing: Boolean = false) {
    val localPrintMissing = printMissing
    val filterF: Annotation => Boolean = g => g != null || localPrintMissing

    gds.exportGenotypes(path, expr, typeFile, filterF)
  }

  /**
    *
    * @param path output path
    * @param append append file to header
    * @param exportPP export Hail PLs as a PP format field
    * @param parallel export VCF in parallel using the path argument as a directory
    */
  def exportVCF(path: String, append: Option[String] = None, exportPP: Boolean = false, parallel: Boolean = false) {
    ExportVCF(gds, path, append, exportPP, parallel)
  }

  /**
    *
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype annotation), which returns a boolean value
    * @param keep keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): GenericDataset = {

    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, gds.vaSignature),
      "s" -> (2, TString),
      "sa" -> (3, gds.saSignature),
      "g" -> (4, gds.genotypeSignature),
      "global" -> (5, gds.globalSignature))


    val ec = EvalContext(symTab)
    ec.set(5, gds.globalAnnotation)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val localKeep = keep
    gds.mapValuesWithAll(
      (v: Variant, va: Annotation, s: String, sa: Annotation, g: Annotation) => {
        ec.setAll(v, va, s, sa, g)

        if (Filter.boxedKeepThis(f(), localKeep))
          g
        else
          null
      })
  }

  def persist(storageLevel: String = "MEMORY_AND_DISK"): GenericDataset = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    gds.copy[M](rdd = gds.rdd.persist(level))
  }

  def queryGA(code: String): (Type, Querier) = {

    val st = Map(Annotation.GENOTYPE_HEAD -> (0, gds.genotypeSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def toVDS: VariantDataset = {
    if (gds.genotypeSignature != TGenotype)
      fatal(s"Cannot convert a GDS to a VDS with signature `${ gds.genotypeSignature.toPrettyString() }'")

    gds.mapValues[GenotypeMatrixT](a => a.asInstanceOf[Genotype]).copy[GenotypeMatrixT](isGenericGenotype = false)
  }

  def write(dirname: String, overwrite: Boolean = false): Unit = {
    require(dirname.endsWith(".vds"), "generic dataset write paths must end in '.vds'")
    require(gds.isGenericGenotype, "Can only write datasets with generic genotypes.")

    if (overwrite)
      gds.hadoopConf.delete(dirname, recursive = true)
    else if (gds.hadoopConf.exists(dirname))
      fatal(s"file already exists at `$dirname'")

    gds.writeMetadata(dirname, parquetGenotypes = false)

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
      StructField("variant", Variant.sparkSchema, nullable = false),
      StructField("annotations", gds.vaSignature.schema),
      StructField("gs", ArrayType(gds.genotypeSignature.schema))
    ))
  }

  def summarize(): SummaryResult = {
    gds.rdd
      .aggregate(new SummaryCombiner[Annotation](_.count(_ != null)))(_.merge(_), _.merge(_))
      .result(gds.nSamples)
  }
}
