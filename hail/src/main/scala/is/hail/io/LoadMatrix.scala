package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{ExecuteContext, MatrixIR, MatrixLiteral, MatrixValue, TableLiteral, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.physical.PArray
import is.hail.expr.types.virtual._
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.io.fs.FS
import org.apache.spark.sql.Row

import scala.io.Source
import scala.language.{existentials, implicitConversions}

object LoadMatrix {

  def warnDuplicates(ids: Array[String]) {
    val duplicates = ids.counter().filter(_._2 > 1)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.size } duplicate ${ plural(duplicates.size, "sample ID") }:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
    }
  }

  def parseHeader(fs: FS, file: String, sep: Char, nRowFields: Int, noHeader: Boolean): (Array[String], Int) = {
    if (noHeader) {
      val nCols = fs.readFile(file) { s => Source.fromInputStream(s).getLines().next() }.count(_ == sep) + 1
      (Array(), nCols - nRowFields)
    } else {
      val lines = fs.readFile(file) { s => Source.fromInputStream(s).getLines().take(2).toArray }
      lines match {
        case Array(header, first) =>
          val nCols = first.split(charRegex(sep), -1).length - nRowFields
          if (nCols < 0)
            fatal(s"More row fields ($nRowFields) than columns (${ nRowFields + nCols }) in file: $file")
          (header.split(charRegex(sep), -1), nCols)
        case _ =>
          fatal(s"file in import_matrix contains no data: $file")
      }
    }
  }

  def splitHeader(cols: Array[String], nRowFields: Int, nColIDs: Int): (Array[String], Array[_]) = {
    if (cols.length == nColIDs) {
      (Array.tabulate(nRowFields)(i => s"f$i"), cols)
    } else if (cols.length == nColIDs + nRowFields) {
      (cols.take(nRowFields), cols.drop(nRowFields))
    } else if (cols.isEmpty) {
      (Array.tabulate(nRowFields)(i => s"f$i"), Array.range(0, nColIDs))
    } else
      fatal(
        s"""Expected file header to contain all $nColIDs column IDs and
            | optionally all $nRowFields row field names: found ${ cols.length } header elements.
           """.stripMargin)
  }

  def makePartitionerFromCounts(partitionCounts: Array[Long], kType: TStruct): (RVDPartitioner, Array[Int]) = {
    var includesStart = true
    val keepPartitions = new ArrayBuilder[Int]()
    val rangeBoundIntervals = partitionCounts.zip(partitionCounts.tail).zipWithIndex.flatMap { case ((s, e), i) =>
      val interval = Interval.orNone(kType.ordering,
        Row(if (includesStart) s else s - 1),
        Row(e - 1),
        includesStart, true)
      includesStart = false
      if (interval.isDefined) keepPartitions += i
      interval
    }
    val ranges = rangeBoundIntervals
    (new RVDPartitioner(Array(kType.fieldNames(0)), kType, ranges), keepPartitions.result())
  }

  def verifyRowFields(fieldNames: Array[String], fieldTypes: Map[String, Type]): TStruct = {
    val headerDups = fieldNames.duplicates()
    if (headerDups.nonEmpty)
      fatal(s"Found following duplicate row fields in header: \n    ${ headerDups.mkString("\n    ") }")

    val fields: Array[(String, Type)] = fieldNames.map { name =>
      fieldTypes.get(name) match {
        case Some(t) => (name, t)
        case None => fatal(
          s"""row field $name not found in provided row_fields dictionary.
             |    expected fields:
             |      ${ fieldNames.mkString("\n      ") }
             |    found fields:
             |      ${ fieldTypes.keys.mkString("\n      ") }
           """.stripMargin)
      }
    }
    TStruct(fields: _*)
  }

  def apply(hc: HailContext,
    files: Array[String],
    rowFields: Map[String, Type],
    keyFields: Array[String],
    cellType: TStruct = TStruct("x" -> TInt64()),
    missingValue: String = "NA",
    nPartitions: Option[Int] = None,
    noHeader: Boolean = false,
    sep: Char = '\t'): MatrixIR = {

    require(cellType.size == 1, "cellType can only have 1 field")

    val nAnnotations = rowFields.size

    assert(rowFields.values.forall { t =>
      t.isOfType(TString()) ||
        t.isOfType(TInt32()) ||
        t.isOfType(TInt64()) ||
        t.isOfType(TFloat32()) ||
        t.isOfType(TFloat64())
    })
    val sc = hc.sc
    val fs = hc.sFS

    if (files.isEmpty)
      fatal("no files specified for import_matrix_table.")

    val (header1, nCols) = parseHeader(fs, files.head, sep, nAnnotations, noHeader)
    val (rowFieldNames, colIDs) = splitHeader(header1, nAnnotations, nCols)

    val rowFieldType: TStruct = verifyRowFields(rowFieldNames, rowFields)

    val header1Bc = hc.backend.broadcast(header1)

    if (!noHeader)
      LoadMatrix.warnDuplicates(colIDs.asInstanceOf[Array[String]])

    val lines = sc.textFilesLines(files, nPartitions.getOrElse(sc.defaultMinPartitions))

    val fileByPartition = lines.partitions.map(p => partitionPath(p))
    val firstPartitions = fileByPartition.scanLeft(0) { (i, file) => if (fileByPartition(i) == file) i else i + 1 }.tail

    val partitionCounts = lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        if (firstPartitions(i) == i) {
          if (!noHeader) {
            val hd1 = header1Bc.value
            val hd = it.next().value.split(sep)
            if (!hd1.sameElements(hd)) {
              if (hd1.length != hd.length) {
                fatal(
                  s"""invalid header: lengths of headers differ.
                     |    ${ hd1.length } elements in ${ files(0) }
                     |    ${ hd.length } elements in ${ fileByPartition(i) }
               """.stripMargin
                )
              }
              hd1.zip(hd).zipWithIndex.foreach { case ((s1, s2), j) =>
                if (s1 != s2) {
                  fatal(
                    s"""invalid header: expected elements to be identical for all input files. Found different elements at position $j.
                       |    ${ files(0) }: $s1
                       |    ${ fileByPartition(i) }: $s2""".
                      stripMargin)
                }
              }
            }
          }
        }
        it
      }.countPerPartition().scanLeft(0L)(_ + _)

    val useIndex = keyFields.isEmpty
    val (rowKey, rowType) =
      if (useIndex)
        (Array("row_id"), TStruct("row_id" -> TInt64()) ++ rowFieldType)
      else (keyFields, rowFieldType)

    if (!keyFields.forall(rowType.fieldNames.contains))
      fatal(
        s"""Some row keys not found in row schema:
           |    '${ keyFields.filter(!rowType.fieldNames.contains(_)).mkString("'\n    '") }'
         """.stripMargin
      )

    val matrixType = MatrixType(
      TStruct.empty(),
      colType = TStruct("col_id" -> (if (noHeader) TInt32() else TString())),
      colKey = Array("col_id"),
      rowType = rowType,
      rowKey = rowKey.toFastIndexedSeq,
      entryType = cellType)

    val tt = matrixType.canonicalTableType
    val rvdType = tt.canonicalRVDType

    val compiledLineParser = new CompiledLineParser(
      matrixType,
      useIndex,
      rowFieldType,
      cellType,
      nCols,
      missingValue,
      sep,
      partitionCounts,
      fileByPartition,
      firstPartitions,
      noHeader)

    val rdd = ContextRDD.weaken[RVDContext](lines.filter(l => l.value.nonEmpty))
      .cmapPartitionsWithIndex(compiledLineParser)

    val rvd = if (useIndex) {
      val (partitioner, keepPartitions) = makePartitionerFromCounts(partitionCounts, rvdType.kType.virtualType)
      RVD(rvdType, partitioner, rdd.subsetPartitions(keepPartitions))
    } else
      RVD.coerce(rvdType, rdd)

    MatrixLiteral(matrixType, rvd, Row(), colIDs.map(x => Row(x)))
  }
}

class CompiledLineParser(
  matrixType: MatrixType,
  rowIdx: Boolean,
  parsableRowFields: TStruct,
  entryType: Type,
  nCols: Int,
  missingValue: String,
  sep: Char,
  partitionCounts: Array[Long],
  fileByPartition: Array[String],
  firstPartitions: Array[Int],
  noHeader: Boolean
) extends ((Int, RVDContext, Iterator[WithContext[String]]) => Iterator[RegionValue]) with Serializable {

  @transient private[this] val fb = new Function4Builder[Region, String, Long, Array[String], Long]
  @transient private[this] val mb = fb.apply_method
  @transient private[this] val fragment = mb.newLocal[String]("currentFragment")
  @transient private[this] val line = fb.arg4
  @transient private[this] val i = mb.newLocal[Int]("i")
  @transient private[this] val j = mb.newLocal[Int]("j")
  @transient private[this] val rvdType = matrixType.canonicalTableType.canonicalRVDType
  private[this] val fieldsPerLine = parsableRowFields.size + nCols
  println(matrixType)
  println(parsableRowFields)
  println(entryType)
  println(rvdType)
  @transient private[this] val srvb = new StagedRegionValueBuilder(mb, rvdType.rowType)
  mb.emit(Code(
    srvb.start(),
    if (rowIdx) Code(srvb.addLong(fb.arg3), srvb.advance()) else Code._empty,
    parseRowFields(),
    parseEntries(),
    srvb.end()))

  private[this] val loadParserOnWorker = fb.result()

  private[this] def missingOr(parse: Code[Unit]): Code[Unit] = {
    fragment.invoke[java.lang.Object, Boolean]("equals", missingValue).mux(
      srvb.setMissing(),
      parse)

  }

  private[this] def parseType(srvb: StagedRegionValueBuilder, t: Type): Code[Unit] = {
    t match {
      case _: TInt32 => missingOr(
        srvb.addInt(
          Code.invokeStatic[java.lang.Integer, String, Int]("parseInt", fragment)))
      case _: TInt64 => missingOr(
        srvb.addLong(
          Code.invokeStatic[java.lang.Long, String, Long]("parseLong", fragment)))
      case _: TFloat32 => missingOr(
        srvb.addFloat(
         Code.invokeStatic[java.lang.Float, String, Float]("parseFloat", fragment)))
      case _: TFloat64 => missingOr(
        srvb.addDouble(
         Code.invokeStatic[java.lang.Double, String, Double]("parseDouble", fragment)))
      case _: TString => missingOr(
        srvb.addString(fragment))
    }
  }

  private[this] def parseRowFields(): Code[_] = {
    val fields = parsableRowFields.fields
    var i = 0
    val ab = new ArrayBuilder[Code[_]]()
    while (i < fields.size) {
      ab += Code(
        fragment := line(i),
        parseType(srvb, fields(i).typ),
        srvb.advance())
      i += 1
    }
    Code.apply(ab.result():_*)
  }

  private[this] def parseEntries(): Code[Unit] = {
    val entryPType = matrixType.entryType.physicalType
    assert(entryPType.fields.size == 1)
    srvb.addArray(PArray(entryPType), { srvb =>
      Code(
        srvb.start(nCols),
        i := 0,
        j := parsableRowFields.fields.size,
        Code.whileLoop(i < nCols,
          fragment := line(j),
          srvb.addBaseStruct(entryPType, { srvb =>
            Code(
              srvb.start(),
              parseType(srvb, matrixType.entryType.fields(0).typ),
              srvb.advance())
          }),
          srvb.advance(),
          i := i + 1,
          j := j + 1))
    })
  }

  def apply(
    partition: Int,
    ctx: RVDContext,
    it: Iterator[WithContext[String]]
  ): Iterator[RegionValue] = {
    val filename = fileByPartition(partition)
    if (firstPartitions(partition) == partition && !noHeader) { it.next() }

    val rv = RegionValue(ctx.region)
    val parse = loadParserOnWorker()
    var index = partitionCounts(partition) - partitionCounts(firstPartitions(partition))
    it.map { x =>
      val line = x.value.split(sep)
      if (line.length != fieldsPerLine) {
        fatal(
          s"""Error parse line $index:
             |    expected $fieldsPerLine row fields and entries but only ${line.length} found.
             |    File: $filename
             |    Line:
             |        ${ line.truncate }""".stripMargin
        )
      }
      try {
        rv.setOffset(
          parse(
            ctx.region,
            filename,
            index,
            line))
      } catch {
        case e: Exception => fatal(
          s"""""Error parse line $index:
               |    File: $filename
               |    Line:
               |        ${ line.truncate }""".stripMargin,
          e)
      }
      index += 1
      rv
    }
  }
}
