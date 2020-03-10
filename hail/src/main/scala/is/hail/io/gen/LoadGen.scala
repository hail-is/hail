package is.hail.io.gen

import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.{ExecuteContext, LowerMatrixIR, MatrixHybridReader, MatrixRead, MatrixReader, MatrixValue, TableRead, TableValue}
import is.hail.expr.types.MatrixType
import is.hail.expr.types.virtual._
import is.hail.io.bgen.LoadBgen
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.{RVD, RVDContext, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import is.hail.io.fs.FS
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.apache.spark.broadcast.Broadcast

case class GenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(Annotation, Iterable[Annotation])])

object LoadGen {
  def apply(
    genFile: String,
    sampleFile: String,
    sc: SparkContext,
    fs: FS,
    rgBc: Option[BroadcastValue[ReferenceGenome]],
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.02,
    chromosome: Option[String] = None,
    contigRecoding: Map[String, String] = Map.empty[String, String],
    skipInvalidLoci: Boolean = false): GenResult = {

    val sampleIds = LoadBgen.readSampleFile(fs, sampleFile)

    LoadVCF.warnDuplicates(sampleIds)

    val nSamples = sampleIds.length

    val rdd = sc.textFileLines(genFile, nPartitions.getOrElse(sc.defaultMinPartitions))
      .flatMap(_.map { l =>
        readGenLine(l, nSamples, tolerance, rgBc.map(_.value), chromosome, contigRecoding, skipInvalidLoci)
      }.value)

    GenResult(genFile, nSamples, rdd.count().toInt, rdd = rdd)
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

    if (skipInvalidLoci && !rg.forall(_.isValidLocus(recodedContig, start.toInt)))
      None
    else {
      val locus = Locus.annotation(recodedContig, start.toInt, rg)
      val alleles = FastIndexedSeq(ref, alt)

      val gp = arr.drop(6 - chrCol).map {
        _.toDouble
      }

      if (gp.length != (3 * nSamples))
        fatal("Number of genotype probabilities does not match 3 * number of samples. If no chromosome column is included, use -c to input the chromosome.")

      val gsb = new ArrayBuilder[Annotation]()

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

      Some(annotations -> gsb.result().toIterable)
    }
  }
}

case class MatrixGENReader(
  files: Array[String],
  sampleFile: String,
  chromosome: Option[String],
  nPartitions: Option[Int],
  tolerance: Double,
  rg: Option[String],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean) extends MatrixHybridReader {

  files.foreach { input =>
    if (!HailContext.get.sFS.stripCodec(input).endsWith(".gen"))
      fatal(s"gen inputs must end in .gen[.bgz], found $input")
  }

  if (files.isEmpty)
    fatal(s"arguments refer to no files: ${ files.mkString(",") }")

  private val referenceGenome = rg.map(ReferenceGenome.getReference)

  referenceGenome.foreach(ref => ref.validateContigRemap(contigRecoding))

  private val samples = LoadBgen.readSampleFile(HailContext.get.sFS, sampleFile)
  private val nSamples = samples.length

  // FIXME: can't specify multiple chromosomes
  private val results = files.map(f => LoadGen(f, sampleFile, HailContext.get.sc, HailContext.sFS, referenceGenome.map(_.broadcast), nPartitions,
    tolerance, chromosome, contigRecoding, skipInvalidLoci))

  private val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
  if (unequalSamples.nonEmpty)
    fatal(
      s"""The following GEN files did not contain the expected number of samples $nSamples:
         |  ${ unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ") }""".stripMargin)

  private val noVariants = results.filter(_.nVariants == 0).map(_.file)
  if (noVariants.nonEmpty)
    fatal(
      s"""The following GEN files contain no variants:
         |  ${ noVariants.mkString("\n  ") })""".stripMargin)

  private val nVariants = results.map(_.nVariants).sum

  info(s"Number of GEN files parsed: ${ results.length }")
  info(s"Number of variants in all GEN files: $nVariants")
  info(s"Number of samples in GEN files: $nSamples")

  def columnCount: Option[Int] = Some(nSamples)

  def partitionCounts: Option[IndexedSeq[Long]] = None

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

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val rdd =
      if (tr.dropRows)
        HailContext.get.sc.emptyRDD[(Annotation, Iterable[Annotation])]
      else
        HailContext.get.sc.union(results.map(_.rdd))

    val requestedType = tr.typ
    val requestedRowType = requestedType.rowType
    val (requestedEntryType, dropCols) = requestedRowType.fieldOption(LowerMatrixIR.entriesFieldName) match {
      case Some(fd) => fd.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct] -> false
      case None => TStruct.empty -> true
    }

    val localNSamples = nSamples

    val locusType = requestedRowType.fieldOption("locus").map(_.typ)
    val allelesType = requestedRowType.fieldOption("alleles").map(_.typ)
    val rsidType = requestedRowType.fieldOption("rsid").map(_.typ)
    val varidType = requestedRowType.fieldOption("varid").map(_.typ)

    val gtType = requestedEntryType.fieldOption("GT").map(_.typ)
    val gpType = requestedEntryType.fieldOption("GP").map(_.typ)

    val localRVDType = tr.typ.canonicalRVDType
    val rvd = RVD.coerce(
      localRVDType,
      ContextRDD.weaken(rdd).cmapPartitions { (ctx, it) =>
        val region = ctx.region
        val rvb = new RegionValueBuilder(region)
        val rv = RegionValue(region)

        it.map { case (va, gs) =>

          rvb.start(localRVDType.rowType)
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
          rv.setOffset(rvb.end())
          rv
        }
      },
      ctx)

    val globalValue = makeGlobalValue(ctx, requestedType, samples.map(Row(_)))

    TableValue(tr.typ, globalValue, rvd)
  }
}
