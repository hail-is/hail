package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.{ ExecuteContext, IRParser, MatrixHybridReader, MatrixIR, MatrixLiteral, MatrixValue, TableLiteral, TableRead, TableValue }
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.io.fs.FS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.io.Source
import scala.language.{existentials, implicitConversions}

object TextMatrixReader {

  def warnDuplicates(ids: Array[String]) {
    val duplicates = ids.counter().filter(_._2 > 1)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.size } duplicate ${ plural(duplicates.size, "sample ID") }:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
    }
  }

  def parseHeader(
    fs: FS,
    file: String,
    sep: Char,
    nRowFields: Int,
    hasHeader: Boolean
  ): (Array[String], Int) = {
    if (hasHeader) {
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
    } else {
      val nCols = fs.readFile(file) { s => Source.fromInputStream(s).getLines().next() }.count(_ == sep) + 1
      (Array(), nCols - nRowFields)
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

  def computePartitionCounts(
    lines: RDD[WithContext[String]],
    hasHeader: Boolean,
    firstPartitions: Array[Int],
    paths: Array[String],
    fileByPartition: Array[String],
    header1Bc: BroadcastValue[Array[String]],
    separator: Char
  ): Array[Long] = {
    lines.filter(l => l.value.nonEmpty)
      .mapPartitionsWithIndex { (i, it) =>
        if (firstPartitions(i) == i) {
          if (hasHeader) {
            val hd1 = header1Bc.value
            val hd = it.next().value.split(separator)
            if (!hd1.sameElements(hd)) {
              if (hd1.length != hd.length) {
                fatal(
                  s"""invalid header: lengths of headers differ.
                     |    ${ hd1.length } elements in ${ paths(0) }
                     |    ${ hd.length } elements in ${ fileByPartition(i) }
               """.stripMargin
                )
              }
              hd1.zip(hd).zipWithIndex.foreach { case ((s1, s2), j) =>
                if (s1 != s2) {
                  fatal(
                    s"""invalid header: expected elements to be identical for all input paths. Found different elements at position $j.
                       |    ${ paths(0) }: $s1
                       |    ${ fileByPartition(i) }: $s2""".
                      stripMargin)
                }
              }
            }
          }
        }
        it
      }.countPerPartition().scanLeft(0L)(_ + _)
  }
}

case class TextMatrixReader(
  paths: Array[String],
  nPartitions: Option[Int],
  rowFieldsStr: Map[String, String],
  entryTypeStr: String,
  missingValue: String,
  hasHeader: Boolean,
  separatorStr: String,
  gzipAsBGZip: Boolean,
  addRowId: Boolean
) extends MatrixHybridReader {
  import TextMatrixReader._
  private[this] val hc = HailContext.get
  private[this] val sc = hc.sc
  private[this] val fs = hc.sFS

  assert(separatorStr.length == 1)
  private[this] val separator = separatorStr.charAt(0)
  private[this] val rowFields = rowFieldsStr.mapValues(IRParser.parseType(_))
  private[this] val entryType = TStruct(true, "x" -> IRParser.parseType(entryTypeStr))
  private[this] val resolvedPaths = fs.globAll(paths)
  require(entryType.size == 1, "entryType can only have 1 field")
  if (resolvedPaths.isEmpty)
    fatal("no paths specified for import_matrix_table.")
  assert((rowFields.values ++ entryType.types).forall { t =>
    t.isOfType(TString()) ||
    t.isOfType(TInt32()) ||
    t.isOfType(TInt64()) ||
    t.isOfType(TFloat32()) ||
    t.isOfType(TFloat64())
  })


  private[this] val (header1, nCols) = parseHeader(fs, resolvedPaths.head, separator, rowFields.size, hasHeader)
  private[this] val (rowFieldNames, colIDs) = splitHeader(header1, rowFields.size, nCols)
  if (addRowId && rowFieldNames.contains("row_id")) {
    fatal(
      s"""If no key is specified, `import_matrix_table`, uses 'row_id'
         |as the key, please provide a key or choose a different row field name.\n
         |  Row field names: ${rowFieldNames}""".stripMargin)
  }
  private[this] val rowFieldTypeWithoutRowId = verifyRowFields(rowFieldNames, rowFields)
  private[this] val rowFieldType =
    if (addRowId) TStruct("row_id" -> TInt64()) ++ rowFieldTypeWithoutRowId
    else rowFieldTypeWithoutRowId
  private[this] val header1Bc = hc.backend.broadcast(header1)
  if (hasHeader)
    warnDuplicates(colIDs.asInstanceOf[Array[String]])
  private[this] val lines = HailContext.maybeGZipAsBGZip(gzipAsBGZip) {
    sc.textFilesLines(resolvedPaths, nPartitions.getOrElse(sc.defaultMinPartitions))
  }
  private[this] val fileByPartition = lines.partitions.map(p => partitionPath(p))
  private[this] val firstPartitions = fileByPartition.scanLeft(0) {
    (i, file) => if (fileByPartition(i) == file) i else i + 1 }.tail
  private[this] val _partitionCounts = computePartitionCounts(
    lines,
    hasHeader,
    firstPartitions,
    resolvedPaths,
    fileByPartition,
    header1Bc,
    separator)

  def columnCount = Some(nCols)

  def partitionCounts = Some(_partitionCounts)

  val fullMatrixType = MatrixType(
    TStruct.empty(),
    colType = TStruct("col_id" -> (if (hasHeader) TString() else TInt32())),
    colKey = Array("col_id"),
    rowType = rowFieldType,
    rowKey = Array().toFastIndexedSeq,
    entryType = entryType)

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val requestedType = tr.typ
    val compiledLineParser = new CompiledLineParser(
      rowFieldType,
      requestedType,
      nCols,
      missingValue,
      separator,
      _partitionCounts,
      fileByPartition,
      firstPartitions,
      hasHeader)
    val rdd = ContextRDD.weaken[RVDContext](lines.filter(l => l.value.nonEmpty))
      .cmapPartitionsWithIndex(compiledLineParser)
    val rvd = if (tr.dropRows)
      RVD.empty(sc, requestedType.canonicalRVDType)
    else
      RVD.unkeyed(PStruct.canonical(requestedType.rowType), rdd)
    val globalValue = makeGlobalValue(ctx, requestedType, colIDs.map(Row(_)))
    TableValue(tr.typ, globalValue, rvd)
  }
}

class MatrixParseError(
  val msg: String,
  val filename: String,
  val line: Long,
  val posStart: Int,
  val posEnd: Int
) extends RuntimeException(s"${filename}:${posStart}-${posEnd}, ${msg}")

class CompiledLineParser(
  onDiskRowFieldsType: TStruct,
  requestedTableType: TableType,
  nCols: Int,
  missingValue: String,
  separator: Char,
  partitionCounts: Array[Long],
  fileByPartition: Array[String],
  firstPartitions: Array[Int],
  hasHeader: Boolean
) extends ((Int, RVDContext, Iterator[WithContext[String]]) => Iterator[RegionValue]) with Serializable {
  assert(!missingValue.contains(separator))
  @transient private[this] val requestedRowType = requestedTableType.canonicalPType
  @transient private[this] val entriesType = requestedRowType
    .selfField(MatrixType.entriesIdentifier)
    .map(f => f.typ.asInstanceOf[PArray])
  @transient private[this] val rowFieldsType = requestedRowType
    .dropFields(Set(MatrixType.entriesIdentifier))
  @transient private[this] val fb = new Function4Builder[Region, String, Long, String, Long]("text_matrix_reader")
  @transient private[this] val mb = fb.apply_method
  @transient private[this] val region = fb.arg1
  @transient private[this] val _filename = fb.arg2
  @transient private[this] val _lineNumber = fb.arg3
  @transient private[this] val _line = fb.arg4
  @transient private[this] val filename = mb.newField[String]("filename")
  @transient private[this] val lineNumber = mb.newField[Long]("lineNumber")
  @transient private[this] val line = mb.newField[String]("line")
  @transient private[this] val pos = mb.newField[Int]("pos")
  @transient private[this] val srvb = new StagedRegionValueBuilder(mb, requestedRowType)

  fb.addInitInstructions(Code(
    pos := 0,
    filename := Code._null,
    lineNumber := 0L,
    line := Code._null,
    srvb.init()))


  @transient private[this] val parseStringMb = fb.newMethod[Region, String]
  parseStringMb.emit(parseString(parseStringMb))
  @transient private[this] val parseIntMb = fb.newMethod[Region, Int]
  parseIntMb.emit(parseInt(parseIntMb))
  @transient private[this] val parseLongMb = fb.newMethod[Region, Long]
  parseLongMb.emit(parseLong(parseLongMb))
  @transient private[this] val parseEntriesMbOpt = entriesType.map { entriesType =>
    val parseEntriesMb = fb.newMethod[Region, Unit]
    parseEntriesMb.emit(parseEntries(parseEntriesMb, entriesType))
    parseEntriesMb
  }
  @transient private[this] val parseRowFieldsMb = fb.newMethod[Region, Unit]
  parseRowFieldsMb.emit(parseRowFields(parseRowFieldsMb))

  mb.emit(Code(
    pos := 0,
    filename := _filename,
    lineNumber := _lineNumber,
    line := _line,
    srvb.start(),
    parseRowFieldsMb.invoke(region),
    parseEntriesMbOpt.map(_.invoke(region)).getOrElse(Code._empty),
    srvb.end()))

  private[this] val loadParserOnWorker = fb.result()

  private[this] def parseError[T](msg: Code[String]): Code[T] =
    Code._throw(Code.newInstance[MatrixParseError, String, String, Long, Int, Int](
      msg, filename, lineNumber, pos, pos + 1))

  private[this] def numericValue(c: Code[Char]): Code[Int] =
    ((c < const('0')) || (c > const('9'))).mux(
      parseError(const("invalid character '")
        .concat(c.toS)
        .concat("' in integer literal")),
      (c - const('0')).toI)

  private[this] def endField(p: Code[Int]): Code[Boolean] =
    p.ceq(line.length()) || line(p).ceq(const(separator))

  private[this] def endField(): Code[Boolean] =
    endField(pos)

  private[this] def missingOr(
    mb: MethodBuilder,
    srvb: StagedRegionValueBuilder,
    parse: Code[Unit]
  ): Code[Unit] = {
    assert(missingValue.size > 0)
    val end = mb.newField[Int]
    val condition = Code(
      end := pos + missingValue.size,
      end <= line.length && endField(end) &&
      line.invoke[Int, String, Int, Int, Boolean]("regionMatches",
        pos, missingValue, 0, missingValue.size))
    condition.mux(
      Code(
        pos := end,
        srvb.setMissing()),
      parse)
  }

  private[this] def skipMissingOr(mb: MethodBuilder, skip: Code[Unit]): Code[Unit] = {
    assert(missingValue.size > 0)
    val end = mb.newField[Int]
    val condition = Code(
      end := pos + missingValue.size,
      end < line.length && endField(end) &&
      line.invoke[Int, String, Int, Int, Boolean]("regionMatches",
        pos, missingValue, 0, missingValue.size))
    condition.mux(
      pos := end,
      skip)
  }

  private[this] def parseInt(mb: MethodBuilder): Code[Int] = {
    val mul = mb.newLocal[Int]("mul")
    val v = mb.newLocal[Int]("v")
    val c = mb.newLocal[Char]("c")
    endField().mux(
      parseError("empty integer literal"),
      Code(
        mul := 1,
        (line(pos).ceq(const('-'))).mux(
          Code(
            mul := -1,
            pos := pos + 1),
          Code._empty),
        c := line(pos),
        v := numericValue(c),
        pos := pos + 1,
        Code.whileLoop(!endField(),
          c := line(pos),
          v := v * 10 + numericValue(c),
          pos := pos + 1),
        v * mul))
  }

  private[this] def parseLong(mb: MethodBuilder): Code[Long] = {
    val mul = mb.newLocal[Long]("mulL")
    val v = mb.newLocal[Long]("vL")
    val c = mb.newLocal[Char]("c")
    endField().mux(
      parseError(const("empty long literal at ")),
      Code(
        mul := 1L,
        (line(pos).ceq(const('-'))).mux(
          mul := -1L,
          pos := pos + 1),
        c := line(pos),
        v := numericValue(c).toL,
        pos := pos + 1,
        Code.whileLoop(!endField(),
          c := line(pos),
          v := v * 10L + numericValue(c).toL,
          pos := pos + 1),
        v * mul))
  }

  private[this] def parseString(mb: MethodBuilder): Code[String] = {
    val start = mb.newLocal[Int]("start")
    Code(
      start := pos,
      Code.whileLoop(!endField(),
        pos := pos + 1),
      line.invoke[Int, Int, String]("substring", start, pos))
  }

  private[this] def parseType(mb: MethodBuilder, srvb: StagedRegionValueBuilder, t: PType): Code[Unit] = {
    val parseDefinedValue = t match {
      case _: PInt32 =>
        srvb.addInt(parseIntMb.invoke(region))
      case _: PInt64 =>
        srvb.addLong(parseLongMb.invoke(region))
      case _: PFloat32 =>
        srvb.addFloat(
          Code.invokeStatic[java.lang.Float, String, Float]("parseFloat", parseStringMb.invoke(region)))
      case _: PFloat64 =>
        srvb.addDouble(
          Code.invokeStatic[java.lang.Double, String, Double]("parseDouble", parseStringMb.invoke(region)))
      case _: PString =>
        srvb.addString(parseStringMb.invoke(region))
    }
    if (t.required) parseDefinedValue else missingOr(mb, srvb, parseDefinedValue)
  }

  private[this] def skipType(mb: MethodBuilder, t: PType): Code[Unit] = {
    val skipDefinedValue = Code.whileLoop(!endField(), pos := pos + 1)
    if (t.required) skipDefinedValue else skipMissingOr(mb, skipDefinedValue)
  }

  private[this] def parseRowFields(mb: MethodBuilder): Code[_] = {
    var inputIndex = 0
    var outputIndex = 0
    assert(onDiskRowFieldsType.size >= rowFieldsType.size)
    val ab = new ArrayBuilder[Code[_]]()
    while (inputIndex < onDiskRowFieldsType.size) {
      val onDiskField = onDiskRowFieldsType.fields(inputIndex)
      val onDiskPType = PType.canonical(onDiskField.typ) // will always be optional
      val requestedField =
        if (outputIndex < rowFieldsType.size)
          rowFieldsType.fields(outputIndex)
        else
          null
      if (requestedField == null || onDiskField.name != requestedField.name) {
        if (onDiskField.name != "row_id") {
          ab += Code(
            skipType(mb, onDiskPType),
            pos := pos + 1)
        }
      } else {
        assert(onDiskPType == requestedField.typ)
        val parseAndAddField =
          if (onDiskField.name == "row_id") srvb.addLong(lineNumber)
          else Code(
            parseType(mb, srvb, onDiskPType),
            pos := pos + 1)
        ab += (pos < line.length).mux(
          parseAndAddField,
          parseError(
            const("unexpected end of line while reading row field ")
              .concat(onDiskField.name)))
        ab += srvb.advance()
        outputIndex += 1
      }
      inputIndex += 1
    }
    assert(outputIndex == rowFieldsType.size)
    Code.apply(ab.result():_*)
  }

  private[this] def parseEntries(mb: MethodBuilder, entriesType: PArray): Code[Unit] = {
    val i = mb.newLocal[Int]("i")
    val entryType = entriesType.elementType.asInstanceOf[PStruct]
    assert(entryType.fields.size == 1)
    srvb.addArray(entriesType, { srvb =>
      Code(
        srvb.start(nCols),
        i := 0,
        Code.whileLoop(i < nCols,
          srvb.addBaseStruct(entryType, { srvb =>
            Code(
              srvb.start(),
              (pos >= line.length).mux(
                parseError(
                  const("unexpected end of line while reading entry ")
                    .concat(i.toS)),
                Code._empty),
              parseType(mb, srvb, entryType.fields(0).typ),
              pos := pos + 1,
              srvb.advance())
          }),
          srvb.advance(),
          i := i + 1))
    })
  }

  def apply(
    partition: Int,
    ctx: RVDContext,
    it: Iterator[WithContext[String]]
  ): Iterator[RegionValue] = {
    val filename = fileByPartition(partition)
    if (firstPartitions(partition) == partition && hasHeader) { it.next() }

    val rv = RegionValue(ctx.region)
    val parse = loadParserOnWorker()
    var index = partitionCounts(partition) - partitionCounts(firstPartitions(partition))
    it.map { x =>
      try {
        rv.setOffset(
          parse(
            ctx.region,
            filename,
            index,
            x.value))
      } catch {
        case e: MatrixParseError =>
          fatal(
          s"""""Error parse line $index:${e.posStart}-${e.posEnd}:
               |    File: $filename
               |    Line:
               |        ${ x.value.truncate }""".stripMargin,
          e)
        case e: Exception => fatal(
          s"""""Error parse line $index:
               |    File: $filename
               |    Line:
               |        ${ x.value.truncate }""".stripMargin,
          e)
      }
      index += 1
      rv
    }
  }
}
