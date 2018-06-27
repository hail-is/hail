package is.hail.utils

import java.util.regex.Pattern

import is.hail.HailContext
import is.hail.expr._
import is.hail.expr.ir.TableImport
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils.StringEscapeUtils._
import org.apache.spark.rdd.RDD

import scala.util.matching.Regex

case class TableReaderOptions(
  nPartitions: Int,
  commentStartsWith: Array[String],
  commentRegexes: Array[Regex],
  separator: String,
  missing: String,
  noHeader: Boolean,
  header: String,
  quote: java.lang.Character,
  skipBlankLines: Boolean,
  useColIndices: Array[Int],
  originalType: TStruct) {
  assert(useColIndices.isSorted)

  def isComment(s: String): Boolean = TextTableReader.isCommentLine(commentStartsWith, commentRegexes)(s)
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

  val booleanRegex = """^([Tt]rue)|([Ff]alse)|(TRUE)|(FALSE)$"""
  val doubleRegex = """^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"""
  val intRegex = """^-?\d+$"""

  def isCommentLine(commentStartsWith: Array[String], commentRegexes: Array[Regex])(line: String): Boolean = {
    commentStartsWith.exists(pattern => line.startsWith(pattern)) || commentRegexes.exists(pattern => pattern.matches(line))
  }

  def imputeTypes(values: RDD[WithContext[String]], header: Array[String],
    delimiter: String, missing: String, quote: java.lang.Character): Array[Option[Type]] = {
    val nFields = header.length
    val regexes = Array(booleanRegex, intRegex, doubleRegex).map(Pattern.compile)

    val regexTypes: Array[Type] = Array(TBoolean(), TInt32(), TFloat64())
    val nRegex = regexes.length

    val imputation = values.mapPartitions { it =>
      val ma = MultiArray2.fill[Boolean](nFields, nRegex + 1)(true)
      val ab = new ArrayBuilder[String]
      val sb = new StringBuilder
      it.foreach { line => line.foreach { l =>
        val split = splitLine(l, delimiter, quote, ab, sb)
        if (split.length != nFields)
          fatal(s"expected $nFields fields, but found ${ split.length }")

        var i = 0
        while (i < nFields) {
          val field = split(i)
          if (field != missing) {
            var j = 0
            while (j < nRegex) {
              ma.update(i, j, ma(i, j) && regexes(j).matcher(field).matches())
              j += 1
            }
            ma.update(i, nRegex, false)
          }
          i += 1
        }
      }}
      Iterator.single(ma)
    }
      .reduce({ case (ma1, ma2) =>
      var i = 0
      while (i < nFields) {
        var j = 0
        while (j < nRegex) {
          ma1.update(i, j, ma1(i, j) && ma2(i, j))
          j += 1
        }
        ma1.update(i, nRegex, ma1(i, nRegex) && ma2(i, nRegex))
        i += 1
      }
      ma1
    })

    imputation.rowIndices.map { i =>
      someIf(!imputation(i, nRegex),
        (0 until nRegex).find(imputation(i, _))
          .map(regexTypes)
          .getOrElse(TString()))
    }.toArray
  }

  def read(hc: HailContext)(files: Array[String],
    types: Map[String, Type] = Map.empty[String, Type],
    comment: Array[String] = Array.empty[String],
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false,
    nPartitions: Int = hc.sc.defaultMinPartitions,
    quote: java.lang.Character = null,
    skipBlankLines: Boolean = false): Table = {

    require(files.nonEmpty)

    val commentStartsWith = comment.filter(_.length == 1)
    val commentRegexes = comment.filter(_.length > 1).map(_.r)

    val isComment = isCommentLine(commentStartsWith, commentRegexes) _

    val firstFile = files.head
    val header = hc.hadoopConf.readLines(firstFile) { lines =>
      val filt = lines.filter(line => !isComment(line.value) && !(skipBlankLines && line.value.isEmpty))

      if (filt.isEmpty)
        fatal(
          s"""invalid file: no lines remaining after comment filter
             |  Offending file: $firstFile""".stripMargin)
      else
        filt.next().value
    }

    val splitHeader = splitLine(header, separator, quote)
    val preColumns = if (noHeader) {
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

    val rdd = hc.sc.textFilesLines(files, nPartitions)
      .filter { line =>
        !isComment(line.value) &&
          (noHeader || line.value != header) &&
          !(skipBlankLines && line.value.isEmpty)
      }

    val sb = new StringBuilder

    val namesAndTypes = {
      if (impute) {
        info("Reading table to impute column types")

        sb.append("Finished type imputation")
        val imputedTypes = imputeTypes(rdd, columns, separator, missing, quote)
        columns.zip(imputedTypes).map { case (name, imputedType) =>
          types.get(name) match {
            case Some(t) =>
              sb.append(s"\n  Loading column '$name' as type '$t' (user-specified)")
              (name, t)
            case None =>
              imputedType match {
                case Some(t) =>
                  sb.append(s"\n  Loading column '$name' as type '$t' (imputed)")
                  (name, t)
                case None =>
                  sb.append(s"\n  Loading column '$name' as type 'str' (no non-missing values for imputation)")
                  (name, TString())
              }
          }
        }
      }
      else {
        sb.append("Reading table with no type imputation\n")
        columns.map { c =>
          types.get(c) match {
            case Some(t) =>
              sb.append(s"  Loading column '$c' as type '$t' (user-specified)\n")
              (c, t)
            case None =>
              sb.append(s"  Loading column '$c' as type 'str' (type not specified)\n")
              (c, TString())
          }
        }
      }
    }

    info(sb.result())

    val ttyp = TableType(TStruct(namesAndTypes: _*), None, TStruct())
    val readerOpts = TableReaderOptions(nPartitions, commentStartsWith, commentRegexes,
      separator, missing, noHeader, header, quote, skipBlankLines, namesAndTypes.indices.toArray,
      ttyp.rowType)
    new Table(hc, TableImport(files, ttyp, readerOpts))
  }
}
