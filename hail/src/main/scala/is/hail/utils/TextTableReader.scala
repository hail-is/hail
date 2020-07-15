package is.hail.expr.ir

import java.util.regex.Pattern

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.lowering.TableStage
import is.hail.io.fs.{FS, FileStatus}
import is.hail.rvd.RVDPartitioner
import is.hail.types._
import is.hail.types.physical.{PCanonicalStruct, PStruct, PType}
import is.hail.types.virtual._
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JValue}

import scala.collection.mutable
import scala.util.matching.Regex

abstract class TextReaderOptions {
  val comment: Array[String]
  val hasHeader: Boolean

  private lazy val commentStartsWith: Array[String] = comment.filter(_.length == 1)
  private lazy val commentRegexes: Array[Regex] = comment.filter(_.length > 1).map(_.r)

  final def isComment(line: String): Boolean =
    commentStartsWith.exists(pattern => line.startsWith(pattern)) || commentRegexes.exists(pattern => pattern.matches(line))
}

case class TextTableReaderParameters(
  files: Array[String],
  typeMapStr: Map[String, String],
  comment: Array[String],
  separator: String,
  missing: Set[String],
  hasHeader: Boolean,
  nPartitionsOpt: Option[Int],
  quoteStr: String,
  skipBlankLines: Boolean,
  forceBGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  forceGZ: Boolean) extends TextReaderOptions {
  @transient val typeMap: Map[String, Type] = typeMapStr.mapValues(s => IRParser.parseType(s)).map(identity)

  val quote: java.lang.Character = if (quoteStr != null) quoteStr(0) else null

  def nPartitions: Int = nPartitionsOpt.getOrElse(HailContext.backend.defaultParallelism)
}

case class TextTableReaderMetadata(fileStatuses: Array[FileStatus], header: String, rowPType: PStruct) {
  def fullType: TableType = TableType(rowType = rowPType.virtualType, globalType = TStruct(), key = FastIndexedSeq())
}

object TextTableReader {

  def splitLine(s: String, separator: String, quote: java.lang.Character): Array[String] =
    splitLine(s, separator, quote, new ArrayBuilder[String], new StringBuilder)

  def splitLine(
    s: String,
    separator: String,
    quote: java.lang.Character,
    ab: ArrayBuilder[String],
    sb: StringBuilder): Array[String] = {
    ab.clear()
    sb.clear()

    val matchSep: Int => Int = separator.length match {
      case 0 => fatal("Hail does not currently support 0-character separators")
      case 1 =>
        val sepChar = separator(0)
        (i: Int) => if (s(i) == sepChar) 1 else -1
      case _ =>
        val p = Pattern.compile(separator)
        val m = p.matcher(s)

      { (i: Int) =>
        m.region(i, s.length)
        if (m.lookingAt())
          m.end() - m.start()
        else
          -1
      }
    }

    var i = 0
    while (i < s.length) {
      val c = s(i)

      val l = matchSep(i)
      if (l != -1) {
        i += l
        ab += sb.result()
        sb.clear()
      } else if (quote != null && c == quote) {
        if (sb.nonEmpty)
          fatal(s"opening quote character '$quote' not at start of field")
        i += 1 // skip quote

        while (i < s.length && s(i) != quote) {
          sb += s(i)
          i += 1
        }

        if (i == s.length)
          fatal(s"missing terminating quote character '$quote'")
        i += 1 // skip quote

        // full field must be quoted
        if (i < s.length) {
          val l = matchSep(i)
          if (l == -1)
            fatal(s"terminating quote character '$quote' not at end of field")
          i += l
          ab += sb.result()
          sb.clear()
        }
      } else {
        sb += c
        i += 1
      }
    }
    ab += sb.result()

    ab.result()
  }

  type Matcher = String => Boolean
  val booleanMatcher: Matcher = x => try {
    x.toBoolean
    true
  } catch {
    case e: IllegalArgumentException => false
  }
  val int32Matcher: Matcher = x => try {
    Integer.parseInt(x)
    true
  } catch {
    case e: NumberFormatException => false
  }
  val int64Matcher: Matcher = x => try {
    java.lang.Long.parseLong(x)
    true
  } catch {
    case e: NumberFormatException => false
  }
  val float64Matcher: Matcher = x => try {
    java.lang.Double.parseDouble(x)
    true
  } catch {
    case e: NumberFormatException => false
  }

  def imputeTypes(
    fs: FS,
    fileStatuses: Array[FileStatus],
    params: TextTableReaderParameters,
    headerLine: String,
    columns: Array[String],
    delimiter: String,
    missing: Set[String],
    quote: java.lang.Character
  ): Array[(Option[Type], Boolean)] = {
    val nFields = columns.length

    val matchTypes: Array[Type] = Array(TBoolean, TInt32, TInt64, TFloat64)
    val matchers: Array[String => Boolean] = Array(
      booleanMatcher,
      int32Matcher,
      int64Matcher,
      float64Matcher)
    val nMatchers = matchers.length

    val lines = GenericLines.read(fs, fileStatuses, nPartitions = params.nPartitionsOpt,
      blockSizeInMB = None, minPartitions = None, gzAsBGZ = params.forceBGZ, allowSerialRead = params.forceGZ)

    val linesRDD: RDD[GenericLine] = lines.toRDD()

    val (imputation, allDefined) = linesRDD.mapPartitions { it =>
      val allDefined = Array.fill(nFields)(true)
      val ma = MultiArray2.fill[Boolean](nFields, nMatchers + 1)(true)
      val ab = new ArrayBuilder[String]
      val sb = new StringBuilder
      it.foreach { genericLine =>
        val line = genericLine.toString

        if (!params.isComment(line) &&
          (!params.hasHeader || line != headerLine) &&
          !(params.skipBlankLines && line.isEmpty)) {

          try {
            val split = splitLine(line, delimiter, quote, ab, sb)
            if (split.length != nFields)
              fatal(s"expected $nFields fields, but found ${ split.length }")

            var i = 0
            while (i < nFields) {
              val field = split(i)
              if (!missing.contains(field)) {
                var j = 0
                while (j < nMatchers) {
                  ma.update(i, j, ma(i, j) && matchers(j)(field))
                  j += 1
                }
                ma.update(i, nMatchers, false)
              } else
                allDefined(i) = false
              i += 1
            }
          } catch {
            case e: Throwable =>
              fatal(
                s"""Caught exception while reading ${ genericLine.file }: ${ e.getMessage }
                   |  offending line: @1""".stripMargin, line, e)
          }
        }
      }
      Iterator.single((ma, allDefined))
    }
      .reduce({ case ((ma1, allDefined1), (ma2, allDefined2)) =>
        var i = 0
        while (i < nFields) {
          var j = 0
          while (j < nMatchers) {
            ma1.update(i, j, ma1(i, j) && ma2(i, j))
            j += 1
          }
          ma1.update(i, nMatchers, ma1(i, nMatchers) && ma2(i, nMatchers))
          i += 1
        }
        (ma1, Array.tabulate(allDefined1.length)(i => (allDefined1(i) && allDefined2(i))))
      })

    imputation.rowIndices.map { i =>
      someIf(!imputation(i, nMatchers),
        (0 until nMatchers).find(imputation(i, _))
          .map(matchTypes)
          .getOrElse(TString))
    }.zip(allDefined).toArray
  }

  def readMetadata(fs: FS, options: TextTableReaderParameters): TextTableReaderMetadata = {
    val TextTableReaderParameters(files, _, _, separator, missing, hasHeader, _, _, skipBlankLines, forceBGZ, filterAndReplace, forceGZ) = options

    val fileStatuses: Array[FileStatus] = {
      val status = fs.globAllStatuses(files)
      if (status.isEmpty)
        fatal("arguments refer to no files")
      if (!forceBGZ) {
        status.foreach { status =>
          val file = status.getPath
          if (file.endsWith(".gz"))
            checkGzippedFile(fs, file, forceGZ, forceBGZ)
        }
      }
      status
    }

    val types = options.typeMap
    val quote = options.quote

    val firstFile = fileStatuses.head.getPath
    val header = fs.readLines(firstFile, filterAndReplace) { lines =>
      val filt = lines.filter(line => !options.isComment(line.value) && !(skipBlankLines && line.value.isEmpty))

      if (filt.isEmpty)
        fatal(
          s"""invalid file: no lines remaining after comment filter
             |  Offending file: $firstFile""".stripMargin)
      else
        filt.next().value
    }

    val splitHeader = splitLine(header, separator, quote)
    val preColumns = if (!hasHeader) {
      splitHeader
        .indices
        .map(i => s"f$i")
        .toArray
    } else splitHeader.map(unescapeString)

    val (columns, duplicates) = mangle(preColumns)
    if (duplicates.nonEmpty) {
      warn(s"Found ${ duplicates.length } duplicate ${ plural(duplicates.length, "column") }. Mangled columns follows:\n  @1",
        duplicates.map { case (pre, post) => s"'$pre' -> '$post'" }.truncatable("\n  "))
    }

    val namesAndTypes =
      columns.map { c =>
        types.get(c) match {
          case Some(t) =>
            (c, PType.canonical(t))
          case None =>
            (c, PType.canonical(TString))
        }
      }
    TextTableReaderMetadata(fileStatuses, header, PCanonicalStruct(true, namesAndTypes: _*))
  }

  def apply(fs: FS, params: TextTableReaderParameters): TextTableReader = {
    val metadata = TextTableReader.readMetadata(fs, params)
    new TextTableReader(params, metadata.header, metadata.fileStatuses, metadata.rowPType)
  }

  def fromJValue(fs: FS, jv: JValue): TextTableReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[TextTableReaderParameters]
    TextTableReader(fs, params)
  }
}

class TextTableReader(
  val params: TextTableReaderParameters,
  header: String,
  fileStatuses: IndexedSeq[FileStatus],
  fullRowPType: PStruct
) extends TableReader {
  val fullType: TableType = TableType(fullRowPType.virtualType, FastIndexedSeq.empty, TStruct())

  def pathsUsed: Seq[String] = params.files

  val partitionCounts: Option[IndexedSeq[Long]] = None

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    PType.canonical(requestedType.rowType, required = true).asInstanceOf[PStruct] ->
      PCanonicalStruct.empty(required = true)
  }

  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val fs = ctx.fs

    val lines = GenericLines.read(fs, fileStatuses, nPartitions = params.nPartitionsOpt,
      blockSizeInMB = None, minPartitions = None, gzAsBGZ = params.forceBGZ, allowSerialRead = params.forceGZ)
    val partitioner: Option[RVDPartitioner] = None
    val globals: TStruct => Row = _ => Row.empty

    val localParams = params
    val localHeader = header
    val localFullRowType = fullRowPType
    val bodyPType: TStruct => PStruct = (requestedRowType: TStruct) => localFullRowType.subsetTo(requestedRowType).asInstanceOf[PStruct]
    val linesBody = lines.body
    val nFieldOrig = localFullRowType.size

    val transformer = localParams.filterAndReplace.transformer()
    val body = { (requestedRowType: TStruct) =>
      val useColIndices = requestedRowType.fieldNames.map(localFullRowType.virtualType.fieldIdx)
      val rowFields = requestedRowType.fields.toArray
      val requestedPType = bodyPType(requestedRowType)

      { (region: Region, context: Any) =>

        val rvb = new RegionValueBuilder(region)
        val ab = new ArrayBuilder[String]
        val sb = new StringBuilder
        linesBody(context)
          .filter { bline =>
            val line = transformer(bline.toString)
            if (line == null || localParams.isComment(line) ||
              (localParams.hasHeader && localHeader == line) ||
              (localParams.skipBlankLines && line.isEmpty))
              false
            else {
              try {
                val sp = TextTableReader.splitLine(line, localParams.separator, localParams.quote, ab, sb)
                if (sp.length != nFieldOrig)
                  fatal(s"expected $nFieldOrig fields, but found ${ sp.length } fields")

                rvb.start(requestedPType)
                rvb.startStruct()

                var i = 0
                while (i < useColIndices.length) {
                  val f = rowFields(i)
                  val name = f.name
                  val typ = f.typ
                  val field = sp(useColIndices(i))
                  try {
                    if (localParams.missing.contains(field))
                      rvb.setMissing()
                    else
                      rvb.addAnnotation(typ, TableAnnotationImpex.importAnnotation(field, typ))
                  } catch {
                    case e: Exception =>
                      fatal(s"""${ e.getClass.getName }: could not convert "$field" to $typ in column "$name" """, e)
                  }
                  i += 1
                }

                rvb.endStruct()
                rvb.end()
                true
              } catch {
                case e: Throwable =>
                  fatal(
                    s"""Caught exception while reading ${ bline.file }: ${ e.getMessage }
                       |  offending line: @1""".stripMargin, line, e)
              }
            }
          }.map(_ => rvb.result().offset)
      }
    }
    new GenericTableValue(partitioner = partitioner,
      fullTableType = fullType,
      globals = globals,
      contextType = lines.contextType,
      contexts = lines.contexts,
      bodyPType = bodyPType,
      body = body)
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType)

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue =
    executeGeneric(ctx).toTableValue(ctx, tr.typ)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "TextTableReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: TextTableReader => params == that.params
    case _ => false
  }
}