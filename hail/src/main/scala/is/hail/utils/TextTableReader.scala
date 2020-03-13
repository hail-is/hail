package is.hail.expr.ir

import java.util.regex.Pattern

import is.hail.HailContext
import is.hail.annotations.{BroadcastRow, RegionValue}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.types._
import is.hail.expr.types.physical.{PCanonicalStruct, PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.rvd.{RVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import org.apache.spark.rdd.RDD

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

case class TextTableReaderOptions(
  files: Array[String],
  typeMapStr: Map[String, String],
  comment: Array[String],
  separator: String,
  missing: Set[String],
  hasHeader: Boolean,
  impute: Boolean,
  nPartitionsOpt: Option[Int],
  quoteStr: String,
  skipBlankLines: Boolean,
  forceBGZ: Boolean,
  filterAndReplace: TextInputFilterAndReplace,
  forceGZ: Boolean) extends TextReaderOptions {
  @transient val typeMap: Map[String, Type] = typeMapStr.mapValues(s => IRParser.parseType(s)).map(identity)

  val quote: java.lang.Character = if (quoteStr != null) quoteStr(0) else null

  def nPartitions: Int = nPartitionsOpt.getOrElse(HailContext.get.sc.defaultParallelism)
}

case class TextTableReaderMetadata(globbedFiles: Array[String], header: String, rowPType: PStruct) {
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

  def imputeTypes(values: RDD[WithContext[String]], header: Array[String],
    delimiter: String, missing: Set[String], quote: java.lang.Character): Array[(Option[Type], Boolean)] = {
    val nFields = header.length

    val matchTypes: Array[Type] = Array(TBoolean, TInt32, TInt64, TFloat64)
    val matchers: Array[String => Boolean] = Array(
      booleanMatcher,
      int32Matcher,
      int64Matcher,
      float64Matcher)
    val nMatchers = matchers.length

    val (imputation, allDefined) = values.mapPartitions { it =>
      val allDefined = Array.fill(nFields)(true)
      val ma = MultiArray2.fill[Boolean](nFields, nMatchers + 1)(true)
      val ab = new ArrayBuilder[String]
      val sb = new StringBuilder
      it.foreach { line =>
        line.foreach { l =>
          val split = splitLine(l, delimiter, quote, ab, sb)
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

  def readMetadata(options: TextTableReaderOptions): TextTableReaderMetadata = {
    HailContext.maybeGZipAsBGZip(options.forceBGZ) {
      readMetadata1(options)
    }
  }

  def readMetadata1(options: TextTableReaderOptions): TextTableReaderMetadata = {
    val hc = HailContext.get

    val TextTableReaderOptions(files, _, comment, separator, missing, hasHeader, impute, _, _, skipBlankLines, forceBGZ, filterAndReplace, forceGZ) = options

    val globbedFiles: Array[String] = {
      val fs = HailContext.get.sFS
      val globbed = fs.globAll(files)
      if (globbed.isEmpty)
        fatal("arguments refer to no files")
      if (!forceBGZ) {
        globbed.foreach { file =>
          if (file.endsWith(".gz"))
            checkGzippedFile(fs, file, forceGZ, forceBGZ)
        }
      }
      globbed
    }

    val types = options.typeMap
    val quote = options.quote
    val nPartitions: Int = options.nPartitions

    val firstFile = globbedFiles.head
    val header = hc.sFS.readLines(firstFile, filterAndReplace) { lines =>
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

    val rdd = hc.sc.textFilesLines(globbedFiles, nPartitions)
      .filter { line =>
        !options.isComment(line.value) &&
          (!hasHeader || line.value != header) &&
          !(skipBlankLines && line.value.isEmpty)
      }

    val sb = new StringBuilder
    val categoryCounts = mutable.Map.empty[String, Int]

    val namesAndTypes = {
      if (impute) {
        info("Reading table to impute column types")

        sb.append("Finished type imputation")
        val imputedTypes = imputeTypes(rdd, columns, separator, missing, quote)
        columns.zip(imputedTypes).map { case (name, (imputedType, req)) =>
          types.get(name) match {
            case Some(t) =>
              sb.append(s"\n  Loading column '$name' as type '$t' (user-specified)")
              categoryCounts.updateValue(s"user-specified $t", 0, _ + 1)
              (name, PType.canonical(t, req))
            case None =>
              imputedType match {
                case Some(t) =>
                  sb.append(s"\n  Loading column '$name' as type '$t' (imputed)")
                  categoryCounts.updateValue(s"imputed $t", 0, _ + 1)
                  (name, PType.canonical(t, req))
                case None =>
                  sb.append(s"\n  Loading column '$name' as type 'str' (no non-missing values for imputation)")
                  categoryCounts.updateValue(s"str (no non-missing values for imputation)", 0, _ + 1)
                  (name, PType.canonical(TString, req))
              }
          }
        }
      } else {
        sb.append("Reading table with no type imputation\n")
        columns.map { c =>
          types.get(c) match {
            case Some(t) =>
              sb.append(s"  Loading column '$c' as type '$t' (user-specified)\n")
              categoryCounts.updateValue(s"user-specified $t", 0, _ + 1)
              (c, PType.canonical(t))
            case None =>
              sb.append(s"  Loading column '$c' as type 'str' (type not specified)\n")
              categoryCounts.updateValue(s"str (type not specified)", 0, _ + 1)
              (c, PType.canonical(TString))
          }
        }
      }
    }

    if (namesAndTypes.length < 50)
      info(sb.result())
    else {
      val countStrs = categoryCounts.toArray
        .sortBy { case (_, n) => -n }
        .map { case (category, n) => s"\n  $n ${ plural(n, "field") }: $category" }
        .mkString("")

      info(s"Loading ${ namesAndTypes.length } fields. Counts by type:$countStrs")
      log.info(sb.result())
    }

    TextTableReaderMetadata(globbedFiles, header, PCanonicalStruct(namesAndTypes: _*))
  }
}

case class TextTableReader(options: TextTableReaderOptions) extends TableReader {
  val partitionCounts: Option[IndexedSeq[Long]] = None

  private lazy val metadata = TextTableReader.readMetadata(options)

  lazy val fullType: TableType = metadata.fullType

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    HailContext.maybeGZipAsBGZip(options.forceBGZ) {
      apply1(tr, ctx)
    }
  }

  def apply1(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val hc = HailContext.get
    val rowTyp = tr.typ.rowType
    val nFieldOrig = fullType.rowType.size
    val rowFields = rowTyp.fields
    val rowPType = PType.canonical(rowTyp).asInstanceOf[PStruct]

    val useColIndices = rowTyp.fields.map(f => fullType.rowType.fieldIdx(f.name))

    val crdd = ContextRDD.textFilesLines(hc.sc, metadata.globbedFiles, options.nPartitions, options.filterAndReplace)
      .filter { line =>
        !options.isComment(line.value) &&
          (!options.hasHeader || metadata.header != line.value) &&
          !(options.skipBlankLines && line.value.isEmpty)
      }.cmapPartitions { (ctx, it) =>
      val region = ctx.region
      val rvb = ctx.rvb
      val rv = RegionValue(region)

      val ab = new ArrayBuilder[String]
      val sb = new StringBuilder
      it.map {
        _.map { line =>
          val sp = TextTableReader.splitLine(line, options.separator, options.quote, ab, sb)
          if (sp.length != nFieldOrig)
            fatal(s"expected $nFieldOrig fields, but found ${ sp.length } fields")

          rvb.set(region)
          rvb.start(rowPType)
          rvb.startStruct()

          var i = 0
          while (i < useColIndices.length) {
            val f = rowFields(i)
            val name = f.name
            val typ = f.typ
            val field = sp(useColIndices(i))
            try {
              if (options.missing.contains(field))
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
          rv.setOffset(rvb.end())
          rv
        }.value
      }
    }

    TableValue(tr.typ, BroadcastRow.empty(ctx), RVD.unkeyed(rowPType, crdd))
  }
}
