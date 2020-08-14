package is.hail.io

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BroadcastValue
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.ir.{EmitFunctionBuilder, EmitMethodBuilder, ExecuteContext, GenericLine, GenericLines, GenericTableValue, IRParser, LowerMatrixIR, MatrixHybridReader, MatrixIR, MatrixLiteral, MatrixValue, TableLiteral, TableRead, TableValue, TextReaderOptions}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.io.fs.{FS, FileStatus}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Extraction, Formats, JObject, JValue}

import scala.collection.mutable
import scala.io.Source
import scala.language.{existentials, implicitConversions}

case class TextMatrixHeaderInfo(
  headerValues: Array[String],
  rowFieldNames: Array[String],
  columnIdentifiers: Array[_] // String or Int
) {
  val nCols: Int = columnIdentifiers.length
}

object TextMatrixReader {

  def warnDuplicates(ids: Array[String]) {
    val duplicates = ids.counter().filter(_._2 > 1)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.size } duplicate ${ plural(duplicates.size, "sample ID") }:\n  @1",
        duplicates.toArray.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
    }
  }

  private def parseHeader(
    fs: FS,
    file: String,
    sep: Char,
    nRowFields: Int,
    opts: TextMatrixReaderOptions
  ): TextMatrixHeaderInfo = {
    val maybeFirstTwoLines = using(fs.open(file)) { s =>
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
        TextMatrixHeaderInfo(
          headerValues,
          headerValues.slice(0, nRowFields),
          headerValues.drop(nRowFields))
      case (true, Seq(header, dataLine)) =>
        val headerValues = header.split(sep)
        val nHeaderValues = headerValues.length
        val nSeparatedValues = dataLine.split(sep).length
        if (nHeaderValues + nRowFields == nSeparatedValues) {
          TextMatrixHeaderInfo(
            headerValues,
            rowFieldNames = Array.tabulate(nRowFields)(i => s"f$i"),
            columnIdentifiers = headerValues)
        } else if (nHeaderValues == nSeparatedValues) {
          TextMatrixHeaderInfo(
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
        TextMatrixHeaderInfo(Array(), Array.tabulate(nRowFields)(i => s"f$i"), Array())
      case (false, firstLine +: _) =>
        val nSeparatedValues = firstLine.split(sep).length
        TextMatrixHeaderInfo(
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

  def checkHeaders(
    header1Path: String,
    header1: Array[String],
    headerPartitions: mutable.Set[Int],
    partitionPaths: Array[String],
    lines: RDD[GenericLine],
    separator: Char
  ): Unit = {
    lines
      .mapPartitionsWithIndex { (i, it) =>
        if (headerPartitions.contains(i)) {
          val hd = it.next().toString.split(separator)
          if (!header1.sameElements(hd)) {
            if (header1.length != hd.length) {
              fatal(
                s"""invalid header: lengths of headers differ.
                   |    ${header1.length} elements in $header1Path
                   |        ${header1.truncate}
                   |    ${hd.length} elements in ${partitionPaths(i)}
                   |        ${hd.truncate}""".stripMargin
              )
            }
            header1.zip(hd).zipWithIndex.foreach { case ((s1, s2), j) =>
              if (s1 != s2) {
                fatal(
                  s"""invalid header: expected elements to be identical for all input paths. Found different elements at position $j.
                     |    ${header1Path}: $s1
                     |    ${partitionPaths(i)}: $s2""".
                    stripMargin)
              }
            }
          }
        }
        it
      }.foreachPartition { _ => () }
  }

  def fromJValue(ctx: ExecuteContext, jv: JValue): TextMatrixReader = {
    val fs = ctx.fs

    implicit val formats: Formats = DefaultFormats
    val params = jv.extract[TextMatrixReaderParameters]

    assert(params.separatorStr.length == 1)
    val separator = params.separatorStr.charAt(0)
    val rowFields = params.rowFieldsStr.mapValues(IRParser.parseType(_))
    val entryType = TStruct("x" -> IRParser.parseType(params.entryTypeStr))
    val fileStatuses = fs.globAllStatuses(params.paths)
    require(entryType.size == 1, "entryType can only have 1 field")
    if (fileStatuses.isEmpty)
      fatal("no paths specified for import_matrix_table.")
    assert((rowFields.values ++ entryType.types).forall { t =>
      t == TString ||
        t == TInt32 ||
        t == TInt64 ||
        t == TFloat32 ||
        t == TFloat64
    })

    val opts = TextMatrixReaderOptions(params.comment, params.hasHeader)

    val headerInfo = parseHeader(fs, fileStatuses.head.getPath, separator, rowFields.size, opts)
    if (params.addRowId && headerInfo.rowFieldNames.contains("row_id")) {
      fatal(
        s"""If no key is specified, `import_matrix_table`, uses 'row_id'
           |as the key, please provide a key or choose a different row field name.\n
           |  Row field names: ${headerInfo.rowFieldNames}""".stripMargin)
    }
    val rowFieldTypeWithoutRowId = verifyRowFields(
      fileStatuses.head.getPath, headerInfo.rowFieldNames, rowFields)
    val rowFieldType =
      if (params.addRowId)
        TStruct("row_id" -> TInt64) ++ rowFieldTypeWithoutRowId
      else
        rowFieldTypeWithoutRowId
    if (params.hasHeader)
      warnDuplicates(headerInfo.columnIdentifiers.asInstanceOf[Array[String]])

    val lines = GenericLines.read(fs, fileStatuses, params.nPartitions, None, None, params.gzipAsBGZip, false)

    val linesRDD = lines.toRDD()
      .filter { line =>
        val l = line.toString
        l.nonEmpty && !opts.isComment(l)
      }

    val linesPartitionCounts = linesRDD.countPerPartition()
    val partitionPaths = lines.contexts.map(a => a.asInstanceOf[Row].getAs[String](1)).toArray

    val headerPartitions = mutable.Set[Int]()
    val partitionLineIndexWithinFile = new Array[Long](linesRDD.getNumPartitions)

    var indexWithinFile = 0L
    var i = 0
    var prevPartitionPath: String = null
    while (i < linesRDD.getNumPartitions) {
      if (linesPartitionCounts(i) > 0) {
        val partPath = partitionPaths(i)
        if (prevPartitionPath == null
          || prevPartitionPath != partPath) {
          prevPartitionPath = partPath
          indexWithinFile = 0
          if (opts.hasHeader) {
            linesPartitionCounts(i) -= 1
            headerPartitions += i
          }
        }
      }
      partitionLineIndexWithinFile(i) = indexWithinFile
      indexWithinFile += linesPartitionCounts(i)
      i += 1
    }

    if (params.hasHeader)
      checkHeaders(fileStatuses.head.getPath, headerInfo.headerValues, headerPartitions, partitionPaths, linesRDD, separator)

    val fullMatrixType = MatrixType(
      TStruct.empty,
      colType = TStruct("col_id" -> (if (params.hasHeader) TString else TInt32)),
      colKey = Array("col_id"),
      rowType = rowFieldType,
      rowKey = Array().toFastIndexedSeq,
      entryType = entryType)

    new TextMatrixReader(params, opts, lines, separator, rowFieldType, fullMatrixType, headerInfo, headerPartitions, linesPartitionCounts, partitionLineIndexWithinFile, partitionPaths)
  }
}

case class TextMatrixReaderParameters(
  paths: Array[String],
  nPartitions: Option[Int],
  rowFieldsStr: Map[String, String],
  entryTypeStr: String,
  missingValue: String,
  hasHeader: Boolean,
  separatorStr: String,
  gzipAsBGZip: Boolean,
  addRowId: Boolean,
  comment: Array[String])

case class TextMatrixReaderOptions(comment: Array[String], hasHeader: Boolean) extends TextReaderOptions

class TextMatrixReader(
  val params: TextMatrixReaderParameters,
  opts: TextMatrixReaderOptions,
  lines: GenericLines,
  separator: Char,
  rowFieldType: TStruct,
  val fullMatrixType: MatrixType,
  headerInfo: TextMatrixHeaderInfo,
  headerPartitions: mutable.Set[Int],
  _partitionCounts: Array[Long],
  partitionLineIndexWithinFile: Array[Long],
  partitionPaths: Array[String]
) extends MatrixHybridReader {
  def pathsUsed: Seq[String] = params.paths

  def columnCount = Some(headerInfo.nCols)

  def partitionCounts = Some(_partitionCounts)

  def rowAndGlobalPTypes(context: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    PType.canonical(requestedType.rowType, required = true).asInstanceOf[PStruct] ->
      PType.canonical(requestedType.globalType, required = true).asInstanceOf[PStruct]
  }

  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val tt = fullMatrixType.toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)

    val globals = Row(headerInfo.columnIdentifiers.map(Row(_)).toFastIndexedSeq)

    val bodyPType = (requestedRowType: TStruct) => PType.canonical(requestedRowType, required = true).asInstanceOf[PStruct]

    val body = { (requestedType: TStruct) =>
      val linesBody = lines.body
      val requestedPType = bodyPType(requestedType)
      val localOpts = opts

      val partitionRowIdxGlobal = (0 until _partitionCounts.length - 1).scanLeft(0L) { case (acc, i) => acc + _partitionCounts(i)}.toArray

      val compiledLineParser = new CompiledLineParser(ctx,
        rowFieldType,
        requestedPType,
        headerInfo.nCols,
        params.missingValue,
        separator,
        headerPartitions,
        _partitionCounts,
        partitionPaths,
        partitionRowIdxGlobal,
        partitionLineIndexWithinFile,
        params.hasHeader)

      { (region: Region, context: Any) =>
        val (lc, partitionIdx: Int) = context
        compiledLineParser.apply(partitionIdx, region,
          linesBody(lc).filter { line =>
            val l = line.toString
            l.nonEmpty && !localOpts.isComment(l)
          }
        )
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
      lines.contexts.zipWithIndex,
      bodyPType,
      body)

  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType)

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    executeGeneric(ctx).toTableValue(ctx,tr.typ)
  }

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "TextMatrixReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: TextMatrixReader => params == that.params
    case _ => false
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
  ctx: ExecuteContext,
  onDiskRowFieldsType: TStruct,
  rowPType: PStruct,
  nCols: Int,
  missingValue: String,
  separator: Char,
  headerPartitions: mutable.Set[Int],
  partitionCounts: Array[Long],
  partitionPaths: Array[String],
  partitionRowIndexGlobal: Array[Long],
  partitionRowIndexFile: Array[Long],
  hasHeader: Boolean
) extends ((Int, Region, Iterator[GenericLine]) => Iterator[Long]) with Serializable {
  assert(!missingValue.contains(separator))
  @transient private[this] val entriesType = rowPType
    .selfField(MatrixType.entriesIdentifier)
    .map(f => f.typ.asInstanceOf[PArray])
  @transient private[this] val rowFieldsType = rowPType
    .dropFields(Set(MatrixType.entriesIdentifier))
  @transient private[this] val fb = EmitFunctionBuilder[Region, String, Long, String, Long](ctx, "text_matrix_reader")
  @transient private[this] val mb = fb.apply_method
  @transient private[this] val region = fb.getCodeParam[Region](1)
  @transient private[this] val _filename = fb.getCodeParam[String](2)
  @transient private[this] val _lineNumber = fb.getCodeParam[Long](3)
  @transient private[this] val _line = fb.getCodeParam[String](4)
  @transient private[this] val filename = mb.genFieldThisRef[String]("filename")
  @transient private[this] val lineNumber = mb.genFieldThisRef[Long]("lineNumber")
  @transient private[this] val line = mb.genFieldThisRef[String]("line")
  @transient private[this] val pos = mb.genFieldThisRef[Int]("pos")
  @transient private[this] val srvb = new StagedRegionValueBuilder(mb, rowPType, region)

  fb.cb.emitInit(Code(
    pos := 0,
    filename := Code._null,
    lineNumber := 0L,
    line := Code._null,
    srvb.init()))


  @transient private[this] val parseStringMb = fb.genEmitMethod[Region, String]("parseString")
  parseStringMb.emit(parseString(parseStringMb))
  @transient private[this] val parseIntMb = fb.genEmitMethod[Region, Int]("parseInt")
  parseIntMb.emit(parseInt(parseIntMb))
  @transient private[this] val parseLongMb = fb.genEmitMethod[Region, Long]("parseLong")
  parseLongMb.emit(parseLong(parseLongMb))
  @transient private[this] val parseRowFieldsMb = fb.genEmitMethod[Region, Unit]("parseRowFields")
  parseRowFieldsMb.emit(parseRowFields(parseRowFieldsMb))

  @transient private[this] val parseEntriesMbOpt = entriesType.map { entriesType =>
    val parseEntriesMb = fb.genEmitMethod[Region, Unit]("parseEntries")
    parseEntriesMb.emit(parseEntries(parseEntriesMb, entriesType))
    parseEntriesMb
  }

  mb.emit(Code(
    pos := 0,
    filename := _filename,
    lineNumber := _lineNumber,
    line := _line,
    srvb.start(),
    parseRowFieldsMb.invokeCode(region),
    parseEntriesMbOpt.map(_.invokeCode(region)).getOrElse(Code._empty),
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
    mb: EmitMethodBuilder[_],
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

  private[this] def skipMissingOr(mb: EmitMethodBuilder[_], skip: Code[Unit]): Code[Unit] = {
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

  private[this] def parseInt(mb: EmitMethodBuilder[_]): Code[Int] = {
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

  private[this] def parseLong(mb: EmitMethodBuilder[_]): Code[Long] = {
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

  private[this] def parseString(mb: EmitMethodBuilder[_]): Code[String] = {
    val start = mb.newLocal[Int]("start")
    Code(
      start := pos,
      Code.whileLoop(!endField(),
        pos := pos + 1),
      line.invoke[Int, Int, String]("substring", start, pos))
  }

  private[this] def parseType(mb: EmitMethodBuilder[_], srvb: StagedRegionValueBuilder, t: PType): Code[Unit] = {
    val parseDefinedValue = t match {
      case _: PInt32 =>
        srvb.addInt(parseIntMb.invokeCode(region))
      case _: PInt64 =>
        srvb.addLong(parseLongMb.invokeCode(region))
      case _: PFloat32 =>
        srvb.addFloat(
          Code.invokeStatic1[java.lang.Float, String, Float]("parseFloat", parseStringMb.invokeCode(region)))
      case _: PFloat64 =>
        srvb.addDouble(
          Code.invokeStatic1[java.lang.Double, String, Double]("parseDouble", parseStringMb.invokeCode(region)))
      case _: PString =>
        srvb.addString(parseStringMb.invokeCode(region))
    }
    if (t.required) parseDefinedValue else missingOr(mb, srvb, parseDefinedValue)
  }

  private[this] def skipType(mb: EmitMethodBuilder[_], t: PType): Code[Unit] = {
    val skipDefinedValue = Code.whileLoop(!endField(), pos := pos + 1)
    if (t.required) skipDefinedValue else skipMissingOr(mb, skipDefinedValue)
  }

  private[this] def parseRowFields(mb: EmitMethodBuilder[_]): Code[_] = {
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

  private[this] def parseEntries(mb: EmitMethodBuilder[_], entriesType: PArray): Code[Unit] = {
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
    r: Region,
    it: Iterator[GenericLine]
  ): Iterator[Long] = {
    val filename = partitionPaths(partition)
    if (hasHeader && headerPartitions.contains(partition))
      it.next()

    val parse = loadParserOnWorker()
    val fileLineIndex = partitionRowIndexFile(partition)
    val globalLineIndex = partitionRowIndexGlobal(partition)

    var idxWithinPartition = 0L
    it.map { line =>
      val x = line.toString
      try {
        val res =
          parse(
            r,
            filename,
            globalLineIndex + idxWithinPartition,
            x)
        idxWithinPartition += 1
        res
      } catch {
        case e: MatrixParseError =>
          fatal(
            s"""""Error parse line ${ fileLineIndex + idxWithinPartition }:${ e.posStart }-${ e.posEnd }:
               |    File: $filename
               |    Line:
               |        ${ x.truncate }""".stripMargin,
            e)
        case e: Exception => fatal(
          s"""""Error parse line ${ fileLineIndex + idxWithinPartition }:
             |    File: $filename
             |    Line:
             |        ${ x.truncate }""".stripMargin,
          e)
      }
    }
  }
}
