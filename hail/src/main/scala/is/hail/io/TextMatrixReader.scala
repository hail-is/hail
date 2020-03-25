package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BroadcastValue
import is.hail.expr.ir.{EmitFunctionBuilder, ExecuteContext, IRParser, MatrixHybridReader, MatrixIR, MatrixLiteral, MatrixValue, TableLiteral, TableRead, TableValue, TextReaderOptions}
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

  private case class HeaderInfo (
    headerValues: Array[String],
    rowFieldNames: Array[String],
    columnIdentifiers: Array[_] // String or Int
  ) {
    val nCols = columnIdentifiers.length
  }

  private def parseHeader(
    fs: FS,
    file: String,
    sep: Char,
    nRowFields: Int,
    opts: TextMatrixReaderOptions
  ): HeaderInfo = {
    val maybeFirstTwoLines = fs.readFile(file) { s =>
      Source.fromInputStream(s).getLines().filter(!opts.isComment(_)).take(2).toArray.toSeq }

    (opts.hasHeader, maybeFirstTwoLines) match {
      case (true, Seq()) =>
        fatal(s"Expected header in every file, but found empty file: $file")
      case (true, Seq(header)) =>
        warn(s"File $file contains a header, but no lines of data.")
        val headerValues = header.split(sep)
        if (headerValues.length < nRowFields) {
          fatal(
            s"""File ${file} contains one line and you told me it had a header,
               |so I expected to see at least the ${nRowFields} row field names
               |on the header line, but instead I only saw ${headerValues.length}
               |separated values. The header was:
               |    ${header}""".stripMargin)
        }
        HeaderInfo(
          headerValues,
          headerValues.slice(0, nRowFields),
          headerValues.drop(nRowFields))
      case (true, Seq(header, dataLine)) =>
        val headerValues = header.split(sep)
        val nHeaderValues = headerValues.length
        val nSeparatedValues = dataLine.split(sep).length
        if (nHeaderValues + nRowFields == nSeparatedValues) {
          HeaderInfo(
            headerValues,
            rowFieldNames = Array.tabulate(nRowFields)(i => s"f$i"),
            columnIdentifiers = headerValues)
        } else if (nHeaderValues == nSeparatedValues) {
          HeaderInfo(
            headerValues,
            rowFieldNames = headerValues.slice(0, nRowFields),
            columnIdentifiers = headerValues.drop(nRowFields))
        } else {
          fatal(
            s"""In file $file, expected the header line to match either:
               |    rowField0 rowField1 ... rowField${nRowFields} colId0 colId1 ...
               |or
               |    colId0 colId1 ...
               |Instead the first two lines were:
               |    ${header.truncate}
               |    ${dataLine.truncate}
               |The first line contained ${nHeaderValues} separated values and the
               |second line contained ${nSeparatedValues} separated values.""".stripMargin)
        }
      case (false, Seq()) =>
        warn(s"File $file is empty and has no header, so we assume no columns.")
        HeaderInfo(Array(), Array.tabulate(nRowFields)(i => s"f$i"), Array())
      case (false, firstLine +: _) =>
        val nSeparatedValues = firstLine.split(sep).length
        HeaderInfo(
          Array(),
          Array.tabulate(nRowFields)(i => s"f$i"),
          Array.range(0, nSeparatedValues - nRowFields))
    }
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

  def verifyRowFields(
    fileName: String,
    fieldNames: Array[String],
    fieldTypes: Map[String, Type]
  ): TStruct = {
    val headerDups = fieldNames.duplicates()
    if (headerDups.nonEmpty)
      fatal(s"Found following duplicate row fields in header: \n    ${ headerDups.mkString("\n    ") }")

    val fields: Array[(String, Type)] = fieldNames.map { name =>
      fieldTypes.get(name) match {
        case Some(t) => (name, t)
        case None =>
          val rowFieldsAsPython = fieldTypes
            .map { case (fieldName, typ) => s"'${fieldName}': ${typ.toString}" }
            .mkString("{", ",\n       ", "}")
          fatal(
          s"""In file $fileName, found a row field, $name, that is not in `row_fields':
             |    row fields found in file:
             |      ${ fieldNames.mkString("\n      ") }
             |    row_fields:
             |      ${ rowFieldsAsPython }
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
                     |        ${hd1.truncate}
                     |    ${ hd.length } elements in ${ fileByPartition(i) }
                     |        ${hd.truncate}""".stripMargin
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
      }.countPerPartition()
  }
}

case class TextMatrixReaderOptions(comment: Array[String], hasHeader: Boolean) extends TextReaderOptions

case class TextMatrixReader(
  paths: Array[String],
  nPartitions: Option[Int],
  rowFieldsStr: Map[String, String],
  entryTypeStr: String,
  missingValue: String,
  hasHeader: Boolean,
  separatorStr: String,
  gzipAsBGZip: Boolean,
  addRowId: Boolean,
  comment: Array[String]
) extends MatrixHybridReader {
  import TextMatrixReader._
  private[this] val hc = HailContext.get
  private[this] val sc = hc.sc
  private[this] val fs = hc.sFS

  assert(separatorStr.length == 1)
  private[this] val separator = separatorStr.charAt(0)
  private[this] val rowFields = rowFieldsStr.mapValues(IRParser.parseType(_))
  private[this] val entryType = TStruct("x" -> IRParser.parseType(entryTypeStr))
  private[this] val resolvedPaths = fs.globAll(paths)
  require(entryType.size == 1, "entryType can only have 1 field")
  if (resolvedPaths.isEmpty)
    fatal("no paths specified for import_matrix_table.")
  assert((rowFields.values ++ entryType.types).forall { t =>
    t == TString ||
    t == TInt32 ||
    t == TInt64 ||
    t == TFloat32 ||
    t == TFloat64
  })

  val opts = TextMatrixReaderOptions(comment, hasHeader)

  private[this] val headerInfo = parseHeader(fs, resolvedPaths.head, separator, rowFields.size, opts)
  if (addRowId && headerInfo.rowFieldNames.contains("row_id")) {
    fatal(
      s"""If no key is specified, `import_matrix_table`, uses 'row_id'
         |as the key, please provide a key or choose a different row field name.\n
         |  Row field names: ${headerInfo.rowFieldNames}""".stripMargin)
  }
  private[this] val rowFieldTypeWithoutRowId = verifyRowFields(
    resolvedPaths.head, headerInfo.rowFieldNames, rowFields)
  private[this] val rowFieldType =
    if (addRowId) TStruct("row_id" -> TInt64) ++ rowFieldTypeWithoutRowId
    else rowFieldTypeWithoutRowId
  private[this] val header1Bc = hc.backend.broadcast(headerInfo.headerValues)
  if (hasHeader)
    warnDuplicates(headerInfo.columnIdentifiers.asInstanceOf[Array[String]])
  private[this] val lines = HailContext.maybeGZipAsBGZip(gzipAsBGZip) {
    val localOpts = opts
    sc.textFilesLines(resolvedPaths, nPartitions.getOrElse(sc.defaultMinPartitions))
      .filter(line => !localOpts.isComment(line.value))
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

  def columnCount = Some(headerInfo.nCols)

  def partitionCounts = Some(_partitionCounts)

  val fullMatrixType = MatrixType(
    TStruct.empty,
    colType = TStruct("col_id" -> (if (hasHeader) TString else TInt32)),
    colKey = Array("col_id"),
    rowType = rowFieldType,
    rowKey = Array().toFastIndexedSeq,
    entryType = entryType)

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val requestedType = tr.typ
    val compiledLineParser = new CompiledLineParser(
      rowFieldType,
      requestedType,
      headerInfo.nCols,
      missingValue,
      separator,
      _partitionCounts,
      fileByPartition,
      firstPartitions,
      hasHeader)
    val rdd = ContextRDD.weaken(lines.filter(l => l.value.nonEmpty))
      .cmapPartitionsWithIndex(compiledLineParser)
    val rvd = if (tr.dropRows)
      RVD.empty(sc, requestedType.canonicalRVDType)
    else
      RVD.unkeyed(PStruct.canonical(requestedType.rowType), rdd)
    val globalValue = makeGlobalValue(ctx, requestedType, headerInfo.columnIdentifiers.map(Row(_)))
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
  @transient private[this] val fb = EmitFunctionBuilder[Region, String, Long, String, Long]("text_matrix_reader")
  @transient private[this] val mb = fb.apply_method
  @transient private[this] val region = fb.getArg[Region](1)
  @transient private[this] val _filename = fb.getArg[String](2)
  @transient private[this] val _lineNumber = fb.getArg[Long](3)
  @transient private[this] val _line = fb.getArg[String](4)
  @transient private[this] val filename = mb.genFieldThisRef[String]("filename")
  @transient private[this] val lineNumber = mb.genFieldThisRef[Long]("lineNumber")
  @transient private[this] val line = mb.genFieldThisRef[String]("line")
  @transient private[this] val pos = mb.genFieldThisRef[Int]("pos")
  @transient private[this] val srvb = new StagedRegionValueBuilder(mb, requestedRowType)

  fb.cb.emitInit(Code(
    pos := 0,
    filename := Code._null,
    lineNumber := 0L,
    line := Code._null,
    srvb.init()))


  @transient private[this] val parseStringMb = fb.genMethod[Region, String]("parseString")
  parseStringMb.emit(parseString(parseStringMb))
  @transient private[this] val parseIntMb = fb.genMethod[Region, Int]("parseInt")
  parseIntMb.emit(parseInt(parseIntMb))
  @transient private[this] val parseLongMb = fb.genMethod[Region, Long]("parseLong")
  parseLongMb.emit(parseLong(parseLongMb))
  @transient private[this] val parseRowFieldsMb = fb.genMethod[Region, Unit]("parseRowFields")
  parseRowFieldsMb.emit(parseRowFields(parseRowFieldsMb))

  @transient private[this] val parseEntriesMbOpt = entriesType.map { entriesType =>
    val parseEntriesMb = fb.genMethod[Region, Unit]("parseEntries")
    parseEntriesMb.emit(parseEntries(parseEntriesMb, entriesType))
    parseEntriesMb
  }

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

  private[this] def parseError[T](msg: Code[String])(implicit tti: TypeInfo[T]): Code[T] =
    Code._throw[MatrixParseError, T](Code.newInstance[MatrixParseError, String, String, Long, Int, Int](
      msg, filename, lineNumber, pos, pos + 1))

  private[this] def numericValue(c: Code[Char]): Code[Int] =
    Code.memoize(c, "clp_numeric_val_c") { c =>
      ((c < const('0')) || (c > const('9'))).mux(
        parseError[Int](const("invalid character '")
          .concat(c.toS)
          .concat("' in integer literal")),
        (c - const('0')).toI)
    }

  private[this] def endField(p: Code[Int]): Code[Boolean] =
    Code.memoize(p, "clp_end_field_p") { p =>
      p.ceq(line.length()) || line(p).ceq(const(separator))
    }

  private[this] def endField(): Code[Boolean] =
    endField(pos)

  private[this] def missingOr(
    mb: MethodBuilder[_],
    srvb: StagedRegionValueBuilder,
    parse: Code[Unit]
  ): Code[Unit] = {
    assert(missingValue.size > 0)
    val end = mb.genFieldThisRef[Int]()
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

  private[this] def skipMissingOr(mb: MethodBuilder[_], skip: Code[Unit]): Code[Unit] = {
    assert(missingValue.size > 0)
    val end = mb.genFieldThisRef[Int]()
    val condition = Code(
      end := pos + missingValue.size,
      end < line.length && endField(end) &&
      line.invoke[Int, String, Int, Int, Boolean]("regionMatches",
        pos, missingValue, 0, missingValue.size))
    condition.mux(
      pos := end,
      skip)
  }

  private[this] def parseInt(mb: MethodBuilder[_]): Code[Int] = {
    val mul = mb.newLocal[Int]("mul")
    val v = mb.newLocal[Int]("v")
    val c = mb.newLocal[Char]("c")
    endField().mux(
      parseError[Int]("empty integer literal"),
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

  private[this] def parseLong(mb: MethodBuilder[_]): Code[Long] = {
    val mul = mb.newLocal[Long]("mulL")
    val v = mb.newLocal[Long]("vL")
    val c = mb.newLocal[Char]("c")
    endField().mux(
      parseError[Long](const("empty long literal at ")),
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

  private[this] def parseString(mb: MethodBuilder[_]): Code[String] = {
    val start = mb.newLocal[Int]("start")
    Code(
      start := pos,
      Code.whileLoop(!endField(),
        pos := pos + 1),
      line.invoke[Int, Int, String]("substring", start, pos))
  }

  private[this] def parseType(mb: MethodBuilder[_], srvb: StagedRegionValueBuilder, t: PType): Code[Unit] = {
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

  private[this] def skipType(mb: MethodBuilder[_], t: PType): Code[Unit] = {
    val skipDefinedValue = Code.whileLoop(!endField(), pos := pos + 1)
    if (t.required) skipDefinedValue else skipMissingOr(mb, skipDefinedValue)
  }

  private[this] def parseRowFields(mb: MethodBuilder[_]): Code[_] = {
    var inputIndex = 0
    var outputIndex = 0
    assert(onDiskRowFieldsType.size >= rowFieldsType.size)
    val ab = new ArrayBuilder[Code[Unit]]()
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
          parseError[Unit](
            const("unexpected end of line while reading row field ")
              .concat(onDiskField.name)))
        ab += srvb.advance()
        outputIndex += 1
      }
      inputIndex += 1
    }
    assert(outputIndex == rowFieldsType.size)
    Code(ab.result())
  }

  private[this] def parseEntries(mb: MethodBuilder[_], entriesType: PArray): Code[Unit] = {
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
                parseError[Unit](
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
