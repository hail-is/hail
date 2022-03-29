package is.hail.io

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitFunctionBuilder, GenericLine, GenericLines, GenericTableValue, IEmitCode, IRParser, IntArrayBuilder, LowerMatrixIR, MatrixHybridReader, TableRead, TableValue, TextReaderOptions}
import is.hail.io.fs.FS
import is.hail.rvd.RVDPartitioner
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SIndexablePointerValue, SStackStruct, SStringPointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.{SCode, SValue}
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JValue}

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
      Source.fromInputStream(s).getLines().filter(!opts.isComment(_)).take(2).toArray.toSeq
    }

    (opts.hasHeader, maybeFirstTwoLines) match {
      case (true, Seq()) =>
        fatal(s"Expected header in every file, but found empty file: $file")
      case (true, Seq(header)) =>
        warn(s"File $file contains a header, but no lines of data.")
        val headerValues = header.split(sep)
        if (headerValues.length < nRowFields) {
          fatal(
            s"""File ${ file } contains one line and you told me it had a header,
               |so I expected to see at least the ${ nRowFields } row field names
               |on the header line, but instead I only saw ${ headerValues.length }
               |separated values. The header was:
               |    ${ header }""".stripMargin)
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
               |    rowField0 rowField1 ... rowField${ nRowFields } colId0 colId1 ...
               |or
               |    colId0 colId1 ...
               |Instead the first two lines were:
               |    ${ header.truncate }
               |    ${ dataLine.truncate }
               |The first line contained ${ nHeaderValues } separated values and the
               |second line contained ${ nSeparatedValues } separated values.""".stripMargin)
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
    val keepPartitions = new IntArrayBuilder()
    val rangeBoundIntervals = partitionCounts.zip(partitionCounts.tail).zipWithIndex.flatMap { case ((s, e), i) =>
      val interval = Interval.orNone(kType.ordering,
        Row(if (includesStart) s else s - 1),
        Row(e - 1),
        includesStart, true)
      includesStart = false
      if (interval.isDefined)
        keepPartitions.add(i)
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
            .map { case (fieldName, typ) => s"'${ fieldName }': ${ typ.toString }" }
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
                   |    ${ header1.length } elements in $header1Path
                   |        ${ header1.truncate }
                   |    ${ hd.length } elements in ${ partitionPaths(i) }
                   |        ${ hd.truncate }""".stripMargin
              )
            }
            header1.zip(hd).zipWithIndex.foreach { case ((s1, s2), j) =>
              if (s1 != s2) {
                fatal(
                  s"""invalid header: expected elements to be identical for all input paths. Found different elements at position $j.
                     |    ${ header1Path }: $s1
                     |    ${ partitionPaths(i) }: $s2""".
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
           |  Row field names: ${ headerInfo.rowFieldNames }""".stripMargin)
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

    val linesRDD = lines.toRDD(fs)
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

    val bodyPType = (requestedRowType: TStruct) => PType.canonical(requestedRowType, required = true).asInstanceOf[PCanonicalStruct]

    val body = { (requestedType: TStruct) =>
      val linesBody = lines.body
      val requestedPType = bodyPType(requestedType)
      val localOpts = opts

      val partitionRowIdxGlobal = (0 until _partitionCounts.length - 1).scanLeft(0L) { case (acc, i) => acc + _partitionCounts(i) }.toArray

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

      { (region: Region, theHailClassLoader: HailClassLoader, fs: FS, context: Any) =>
        val Row(lc, partitionIdx: Int) = context
        compiledLineParser.apply(partitionIdx, region, theHailClassLoader,
          linesBody(fs, lc).filter { line =>
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
      TTuple(lines.contextType, TInt32),
      lines.contexts.zipWithIndex.map { case (x, i) => Row(x, i) },
      bodyPType,
      body)

  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType)

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    executeGeneric(ctx).toTableValue(ctx, tr.typ)
  }

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "TextMatrixReader")
  }

  override def renderShort(): String = defaultRender()

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
) extends RuntimeException(s"${ filename }:${ posStart }-${ posEnd }, ${ msg }")

class CompiledLineParser(
  ctx: ExecuteContext,
  onDiskRowFieldsType: TStruct,
  rowPType: PCanonicalStruct,
  nCols: Int,
  missingValue: String,
  separator: Char,
  headerPartitions: mutable.Set[Int],
  partitionCounts: Array[Long],
  partitionPaths: Array[String],
  partitionRowIndexGlobal: Array[Long],
  partitionRowIndexFile: Array[Long],
  hasHeader: Boolean
) extends ((Int, Region, HailClassLoader, Iterator[GenericLine]) => Iterator[Long]) with Serializable {
  assert(!missingValue.contains(separator))
  @transient private[this] val entriesType = rowPType
    .selfField(MatrixType.entriesIdentifier)
    .map(f => f.typ.asInstanceOf[PCanonicalArray])
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

  fb.cb.emitInit(Code(
    pos := 0,
    filename := Code._null[String],
    lineNumber := 0L,
    line := Code._null[String]))


  @transient private[this] val parseStringMb = fb.genEmitMethod[Region, String]("parseString")
  parseStringMb.emitWithBuilder(parseString(_))
  @transient private[this] val parseIntMb = fb.genEmitMethod[Region, Int]("parseInt")
  parseIntMb.emitWithBuilder(parseInt(_))
  @transient private[this] val parseLongMb = fb.genEmitMethod[Region, Long]("parseLong")
  parseLongMb.emitWithBuilder(parseLong(_))

  @transient private[this] def parseEntriesOpt(cb: EmitCodeBuilder): Option[EmitCode] = entriesType.map { entriesType =>
    val sc = parseEntries(cb, entriesType)
    EmitCode.present(cb.emb, sc)
  }

  mb.emitWithBuilder[Long] { cb =>
    cb.assign(pos, 0)
    cb.assign(filename, _filename)
    cb.assign(lineNumber, _lineNumber)
    cb.assign(line, _line)
    val rowFields = parseRowFields(cb)
    val entries = parseEntriesOpt(cb)
    rowPType.constructFromFields(cb, region, rowFields ++ entries, deepCopy = false).a
  }

  private[this] val loadParserOnWorker = fb.result()

  private[this] def parseError(cb: EmitCodeBuilder, msg: Code[String]): Unit =
    cb += Code._throw[MatrixParseError, Unit](Code.newInstance[MatrixParseError, String, String, Long, Int, Int](
      msg, filename, lineNumber, pos, pos + 1))

  private[this] def numericValue(cb: EmitCodeBuilder, cCode: Code[Char]): Code[Int] = {
    val c = cb.newLocal[Char]("clp_numeric_val_c", cCode)
    cb.ifx(c < const('0') || c > const('9'),
      parseError(cb, const("invalid character '")
        .concat(c.toS)
        .concat("' in integer literal")))
    (c - const('0')).toI
  }

  private[this] def endField(cb: EmitCodeBuilder, p: Value[Int]): Code[Boolean] = {
    p.ceq(line.length()) || line(p).ceq(const(separator))
  }

  private[this] def endField(cb: EmitCodeBuilder): Code[Boolean] =
    endField(cb, pos)

  private[this] def parseOptionalValue(
    cb: EmitCodeBuilder,
    parse: EmitCodeBuilder => SValue
  ): IEmitCode = {
    assert(missingValue.size > 0)
    val end = cb.newLocal[Int]("parse_optional_value_end", pos + missingValue.size)

    val Lmissing = CodeLabel()

    cb.ifx(end <= line.length,
      cb.ifx(endField(cb, end),
        cb.ifx(line.invoke[Int, String, Int, Int, Boolean]("regionMatches",
          pos, missingValue, 0, missingValue.size),
          {
            cb.assign(pos, end)
            cb.goto(Lmissing)
          })))

    val pc = parse(cb)
    val Ldefined = CodeLabel()
    cb.goto(Ldefined)

    IEmitCode(Lmissing, Ldefined, pc, false)
  }

  private[this] def skipOptionalValue(cb: EmitCodeBuilder, skip: EmitCodeBuilder => Unit): Unit = {
    assert(missingValue.size > 0)
    val end = cb.newLocal[Int]("skip_optional_value_end", pos + missingValue.size)

    val Lfinished = CodeLabel()

    cb.ifx(end <= line.length,
      cb.ifx(endField(cb, end),
        cb.ifx(line.invoke[Int, String, Int, Int, Boolean]("regionMatches",
          pos, missingValue, 0, missingValue.size),
          {
            cb.assign(pos, end)
            cb.goto(Lfinished)
          })))

    skip(cb)

    cb.define(Lfinished)
  }

  private[this] def parseInt(cb: EmitCodeBuilder): Code[Int] = {
    cb.ifx(endField(cb), parseError(cb, "empty integer literal"))

    val mul = cb.newLocal[Int]("mul", 1)
    cb.ifx(line(pos).ceq(const('-')), {
      cb.assign(mul, -1)
      cb.assign(pos, pos + 1)
    })
    val c = cb.newLocal[Char]("c", line(pos))
    val v = cb.newLocal[Int]("v", numericValue(cb, c))
    cb.assign(pos, pos + 1)

    cb.whileLoop(!endField(cb), {
      cb.assign(c, line(pos))
      cb.assign(v, v * const(10) + numericValue(cb, c))
      cb.assign(pos, pos + 1)
    })
    v * mul
  }

  private[this] def parseLong(cb: EmitCodeBuilder): Code[Long] = {
    cb.ifx(endField(cb), parseError(cb, "empty integer literal"))

    val mul = cb.newLocal[Long]("mulL", 1L)
    cb.ifx(line(pos).ceq(const('-')), {
      cb.assign(mul, -1L)
      cb.assign(pos, pos + 1)
    })
    val c = cb.newLocal[Char]("cL", line(pos))
    val v = cb.newLocal[Long]("vL", numericValue(cb, c).toL)
    cb.assign(pos, pos + 1)

    cb.whileLoop(!endField(cb), {
      cb.assign(c, line(pos))
      cb.assign(v, v * const(10L) + numericValue(cb, c).toL)
      cb.assign(pos, pos + 1)
    })
    v * mul
  }

  private[this] def parseString(cb: EmitCodeBuilder): Code[String] = {
    val start = cb.newLocal[Int]("start", pos)
    cb.whileLoop(!endField(cb),
      cb.assign(pos, pos + 1))
    line.invoke[Int, Int, String]("substring", start, pos)
  }

  private[this] def parseValueOfType(cb: EmitCodeBuilder, t: PType): IEmitCode = {
    def parseDefinedValue(cb: EmitCodeBuilder): SValue = t match {
      case t: PInt32 =>
        primitive(cb.memoize(cb.invokeCode[Int](parseIntMb, region)))
      case t: PInt64 =>
        primitive(cb.memoize(cb.invokeCode[Long](parseLongMb, region)))
      case t: PFloat32 =>
        primitive(cb.memoize(Code.invokeStatic1[java.lang.Float, String, Float]("parseFloat", cb.invokeCode(parseStringMb, region))))
      case t: PFloat64 =>
        primitive(cb.memoize(Code.invokeStatic1[java.lang.Double, String, Double]("parseDouble", cb.invokeCode(parseStringMb, region))))
      case t: PString =>
        val st = SStringPointer(t)
        st.constructFromString(cb, region, cb.invokeCode[String](parseStringMb, region))
    }
    if (t.required)
      IEmitCode.present(cb, parseDefinedValue(cb))
    else
      parseOptionalValue(cb, parseDefinedValue)
  }

  private[this] def skipValueOfType(cb: EmitCodeBuilder, t: PType): Unit = {
    def skipDefinedValue(cb: EmitCodeBuilder): Unit = {
      cb.whileLoop(!endField(cb), cb.assign(pos, pos + 1))
    }

    if (t.required) skipDefinedValue(cb) else skipOptionalValue(cb, skipDefinedValue)
  }

  private[this] def parseRowFields(cb: EmitCodeBuilder): Array[EmitCode] = {
    assert(onDiskRowFieldsType.size >= rowFieldsType.size)

    // need to be careful to ensure parsing code is directly appended to code builder, not EmitCode block
    val fieldEmitCodes = new Array[EmitCode](rowFieldsType.size)

    onDiskRowFieldsType.fields.foreach { onDiskField =>
      rowPType.selfField(onDiskField.name) match {

        case Some(requestedField) =>
          val reqFieldType = requestedField.typ
          val reqIndex = requestedField.index


          val ec = if (onDiskField.name == "row_id")
            EmitCode.present(cb.emb, primitive(lineNumber))
          else {
            cb.ifx(pos >= line.length,
              parseError(cb, const("unexpected end of line while reading row field ")
              .concat(onDiskField.name)))
            val ev = parseValueOfType(cb, reqFieldType).memoize(cb, s"field_${onDiskField.name}")
            cb.assign(pos, pos + 1)
            ev.load
          }

          fieldEmitCodes(reqIndex) = ec

        case None =>
          if (onDiskField.name != "row_id") {
            skipValueOfType(cb, PType.canonical(onDiskField.typ))  // will always be optional
            cb.assign(pos, pos + 1)
          }
      }
    }
    fieldEmitCodes
  }

  private[this] def parseEntries(cb: EmitCodeBuilder, entriesType: PCanonicalArray): SIndexablePointerValue = {
    val entryType = entriesType.elementType.asInstanceOf[PCanonicalStruct]
    assert(entryType.fields.size == 1)
    val (push, finish) = entriesType.constructFromFunctions(cb, region, nCols, false)

    val i = cb.newLocal[Int]("i", 0)
    cb.whileLoop(i < nCols, {
      cb.ifx(pos >= line.length, parseError(cb, const("unexpected end of line while reading entry ").concat(i.toS)))

      val ec = EmitCode.fromI(cb.emb)(cb => parseValueOfType(cb, entryType.fields(0).typ))
      push(cb, IEmitCode.present(cb, SStackStruct.constructFromArgs(cb, region, entryType.virtualType, ec)))
      cb.assign(pos, pos + 1)
      cb.assign(i, i + 1)
    })
    finish(cb)
  }

  def apply(
    partition: Int,
    r: Region,
    theHailClassLoader: HailClassLoader,
    it: Iterator[GenericLine]
  ): Iterator[Long] = {
    val filename = partitionPaths(partition)
    if (hasHeader && headerPartitions.contains(partition))
      it.next()

    val parse = loadParserOnWorker(theHailClassLoader)
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
