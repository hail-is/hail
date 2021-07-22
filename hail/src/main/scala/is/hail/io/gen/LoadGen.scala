package is.hail.io.gen

import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.BroadcastValue
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.ir.{ExecuteContext, GenericLines, GenericTableValue, LowerMatrixIR, MatrixHybridReader, MatrixRead, MatrixReader, MatrixValue, TableRead, TableValue}
import is.hail.types.{MatrixType, TableType}
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.virtual._
import is.hail.io.bgen.LoadBgen
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.{RVD, RVDContext, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import is.hail.io.fs.FS
import is.hail.io.gen.LoadGen.readGenLine
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.apache.spark.broadcast.Broadcast
import org.json4s.{DefaultFormats, Extraction, Formats, JObject, JValue}

import scala.collection.mutable

case class GenResult(file: String, nSamples: Int, nVariants: Int)

object LoadGen {
  def apply(
    genFile: String,
    sampleFile: String,
    fs: FS,
    rgBc: Option[BroadcastValue[ReferenceGenome]],
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.02,
    chromosome: Option[String] = None,
    contigRecoding: Map[String, String] = Map.empty[String, String],
    skipInvalidLoci: Boolean = false): GenResult = {

    var nVariants = 0
    val sampleIds = LoadBgen.readSampleFile(fs, sampleFile)

    fs.readLines(genFile)(_.foreach { _=> nVariants = 1 + nVariants})
    println(s"nVariants = $nVariants")

    LoadVCF.warnDuplicates(sampleIds)

    val nSamples = sampleIds.length

    GenResult(genFile, nSamples, nVariants)
  }

  def readGenLine(line: String, nSamples: Int,
    tolerance: Double,
    rg: Option[ReferenceGenome],
    chromosome: Option[String] = None,
    contigRecoding: Map[String, String] = Map.empty[String, String],
    skipInvalidLoci: Boolean = false): Option[(Annotation, Iterable[Annotation])] = {

    val arr = line.split("\\s+")
    val chrCol = if (chromosome.isDefined) 1 else 0
    val chr = chromosome.getOrElse(arr(0))
    val varid = arr(1 - chrCol)
    val rsid = arr(2 - chrCol)
    val start = arr(3 - chrCol)
    val ref = arr(4 - chrCol)
    val alt = arr(5 - chrCol)

    val recodedContig = contigRecoding.getOrElse(chr, chr)

    val foo = skipInvalidLoci && !rg.forall(_.isValidLocus(recodedContig, start.toInt))
    println(s"foo: $foo")
    if (foo)
      None
    else {
      val locus = Locus.annotation(recodedContig, start.toInt, rg)
      println(s"locus: $locus")
      val alleles = FastIndexedSeq(ref, alt)

      val gp = arr.drop(6 - chrCol).map {
        _.toDouble
      }

      if (gp.length != (3 * nSamples))
        fatal("Number of genotype probabilities does not match 3 * number of samples. If no chromosome column is included, use -c to input the chromosome.")

      val gsb = new mutable.ArrayBuffer[Annotation]()

      for (i <- gp.indices by 3) {
        val d0 = gp(i)
        val d1 = gp(i + 1)
        val d2 = gp(i + 2)
        val sumDosages = d0 + d1 + d2

        val a =
          if (math.abs(sumDosages - 1.0) <= tolerance) {
            val gp = Array(d0 / sumDosages, d1 / sumDosages, d2 / sumDosages)
            val gt = Genotype.unboxedGTFromLinear(gp)
            Annotation(if (gt != -1) Call2.fromUnphasedDiploidGtIndex(gt) else null, gp: IndexedSeq[Double])
          } else
            null

        gsb += a
      }

      val annotations = Annotation(locus, alleles, rsid, varid)

      Some(annotations -> gsb.result())
    }
  }
}

object MatrixGENReader {
  def fromJValue(ctx: ExecuteContext, jv: JValue): MatrixGENReader = {
    val fs = ctx.fs
    implicit val formats: Formats = DefaultFormats
    val params = jv.extract[MatrixGENReaderParameters]

    params.files.foreach { input =>
      if (!fs.stripCodecExtension(input).endsWith(".gen"))
        fatal(s"gen inputs must end in .gen[.bgz], found $input")
    }

    if (params.files.isEmpty)
      fatal(s"arguments refer to no files: ${ params.files.mkString(",") }")

    val referenceGenome = params.rg.map(ReferenceGenome.getReference)

    referenceGenome.foreach(ref => ref.validateContigRemap(params.contigRecoding))

    val samples = LoadBgen.readSampleFile(fs, params.sampleFile)
    val nSamples = samples.length

    // FIXME: can't specify multiple chromosomes
    val results = params.files.map(f => LoadGen(f, params.sampleFile, fs, referenceGenome.map(_.broadcast), params.nPartitions,
      params.tolerance, params.chromosome, params.contigRecoding, params.skipInvalidLoci))
    println(s"param files: ${params.files}")
    val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
    if (unequalSamples.nonEmpty)
      fatal(
        s"""The following GEN files did not contain the expected number of samples $nSamples:
           |  ${ unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ") }""".stripMargin)

    val noVariants = results.filter(_.nVariants == 0).map(_.file)
    if (noVariants.nonEmpty)
      fatal(
        s"""The following GEN files contain no variants:
           |  ${ noVariants.mkString("\n  ") })""".stripMargin)

    val nVariants = results.map(_.nVariants).sum

    info(s"Number of GEN files parsed: ${ results.length }")
    info(s"Number of variants in all GEN files: $nVariants")
    info(s"Number of samples in GEN files: $nSamples")

    def fullMatrixType: MatrixType = MatrixType(
      globalType = TStruct.empty,
      colKey = Array("s"),
      colType = TStruct("s" -> TString),
      rowKey = Array("locus", "alleles"),
      rowType = TStruct(
        "locus" -> TLocus.schemaFromRG(referenceGenome),
        "alleles" -> TArray(TString),
        "rsid" -> TString, "varid" -> TString),
      entryType = TStruct("GT" -> TCall,
        "GP" -> TArray(TFloat64)))
    println(s"Line 169 in JValue before MatrixGenReader creation")
    new MatrixGENReader(params, referenceGenome.map(_.broadcast), fullMatrixType, samples, results)
  }
}

case class MatrixGENReaderParameters(
  files: Array[String],
  sampleFile: String,
  chromosome: Option[String],
  nPartitions: Option[Int],
  tolerance: Double,
  rg: Option[String],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean)

class MatrixGENReader(
  val params: MatrixGENReaderParameters,
  val rgBc: Option[BroadcastValue[ReferenceGenome]],
  val fullMatrixType: MatrixType,
  samples: Array[String],
  results: Array[GenResult]
) extends MatrixHybridReader {
  def pathsUsed: Seq[String] = params.files

  def nSamples: Int = samples.length

  def columnCount: Option[Int] = Some(nSamples)

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def rowAndGlobalPTypes(context: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    requestedType.canonicalRowPType -> PType.canonical(requestedType.globalType).asInstanceOf[PStruct]
  }

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    println(s"apply line 204 before executeGeneric ")
    executeGeneric(ctx).toTableValue(ctx, tr.typ)

  }
  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val fs = ctx.fs
    val fileStatuses = fs.globAllStatuses(fs.globAll(params.files))

    val localNSamples = nSamples

    val tt = fullMatrixType.toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)

    val lines = GenericLines.read(fs, fileStatuses, params.nPartitions, None, None, false, true)

    val globals = Row(samples.map(Row(_)).toFastIndexedSeq)

    val fullRowPType: PType = tt.canonicalRowPType
    val bodyPType = (requestedRowType: TStruct) =>
      fullRowPType.subsetTo(requestedRowType).asInstanceOf[PStruct]
    val linesBody = lines.body

    val myRgbc = rgBc
    val tolerance = params.tolerance
    val chromosome = params.chromosome
    val contigRecoding = params.contigRecoding
    val skipInvalidLoci = params.skipInvalidLoci


    val body = { (requestedRowType: TStruct) =>

      val requestedPType = bodyPType(requestedRowType)

      val (requestedEntryType, dropCols) = requestedRowType.fieldOption(LowerMatrixIR.entriesFieldName) match {
        case Some(fd) => fd.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct] -> false
        case None => TStruct.empty -> true
      }

      { (region: Region, context: Any) =>
        println(s"line 242 execute generic before rvb creation")
        val rvb = new RegionValueBuilder(region)
        val locusType = requestedRowType.fieldOption("locus").map(_.typ)
        val allelesType = requestedRowType.fieldOption("alleles").map(_.typ)
        val rsidType = requestedRowType.fieldOption("rsid").map(_.typ)
        val varidType = requestedRowType.fieldOption("varid").map(_.typ)
        val gtType = requestedEntryType.fieldOption("GT").map(_.typ)
        val gpType = requestedEntryType.fieldOption("GP").map(_.typ)
        println(s"line 250 execute generic before iteration on linesBody")
        linesBody(context).flatMap { line =>
          println(s"line 250 in executeGeneric")
          val stringLine = line.toString
          val optLine = readGenLine(stringLine, localNSamples, tolerance, myRgbc.map(_.value),
            chromosome, contigRecoding, skipInvalidLoci)
          optLine.map { case (va, gs) =>
            rvb.start(requestedPType)
            rvb.startStruct()
            val Row(locus, alleles, rsid, varid) = va.asInstanceOf[Row]
            locusType.foreach(rvb.addAnnotation(_, locus))
            allelesType.foreach(rvb.addAnnotation(_, alleles))
            rsidType.foreach(rvb.addAnnotation(_, rsid))
            varidType.foreach(rvb.addAnnotation(_, varid))
            if (!dropCols) {
              rvb.startArray(localNSamples)
              gs.foreach {
                case Row(gt, gp) =>
                  rvb.startStruct()
                  gtType.foreach(rvb.addAnnotation(_, gt))
                  gpType.foreach(rvb.addAnnotation(_, gp))
                  rvb.endStruct()
                case null =>
                  rvb.setMissing()
              }
              rvb.endArray()
            }
            rvb.endStruct()

            rvb.end()
          }
        }
      }
    }
    new GenericTableValue(
      tt,
      None,
      { (requestedGlobalsType: Type) =>
        val subset = tt.globalType.valueSubsetter(requestedGlobalsType)
        subset(globals).asInstanceOf[Row]
      },
      lines.contextType,
      lines.contexts,
      bodyPType,
      body)
  }
  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixGENReader")
  }

  def renderShort(): String = defaultRender()

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixGENReader => params == that.params
    case _ => false
  }
}
