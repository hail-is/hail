package is.hail.utils

import java.util.regex.Pattern

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.utils.StringEscapeUtils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object TextTableReader {

  def splitLine(s: String, separator: String, quote: java.lang.Character): Array[String] = {

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

    val ab = new ArrayBuilder[String]
    val sb = new StringBuilder

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
          val l =  matchSep(i)
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
  val variantRegex = """^.+:\d+:[ATGC]+:([ATGC]+|\*)(,([ATGC]+|\*))*$"""
  val locusRegex = """^.+:\d+$"""
  val doubleRegex = """^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"""
  val intRegex = """^-?\d+$"""

  def imputeTypes(values: RDD[WithContext[String]], header: Array[String],
    delimiter: String, missing: String, quote: java.lang.Character): Array[Option[Type]] = {
    val nFields = header.length
    val regexes = Array(booleanRegex, variantRegex, locusRegex, intRegex, doubleRegex).map(Pattern.compile)

    val regexTypes: Array[Type] = Array(TBoolean, TVariant, TLocus, TInt, TDouble)
    val nRegex = regexes.length

    val imputation = values.treeAggregate(MultiArray2.fill[Boolean](nFields, nRegex + 1)(true))({ case (ma, line) =>
      line.foreach { l =>
        val split = splitLine(l, delimiter, quote)
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
      }
      ma
    }, { case (ma1, ma2) =>
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
          .getOrElse(TString))
    }.toArray
  }

  def read(sc: SparkContext)(files: Array[String],
    types: Map[String, Type] = Map.empty[String, Type],
    commentChar: Option[String] = None,
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false,
    nPartitions: Int = sc.defaultMinPartitions,
    quote: java.lang.Character = null): (TStruct, RDD[WithContext[Row]]) = {
    require(files.nonEmpty)

    val firstFile = files.head
    val header = sc.hadoopConfiguration.readLines(firstFile) { lines =>
      val filt = lines
        .filter(line => commentChar.forall(pattern => !line.value.startsWith(pattern)))

      if (filt.isEmpty)
        fatal(
          s"""invalid file: no lines remaining after comment filter
              |  Offending file: $firstFile""".stripMargin)
      else
        filt.next().value
    }

    val splitHeader = splitLine(header, separator, quote)
    val columns = if (noHeader) {
      splitHeader
        .indices
        .map(i => s"f$i")
        .toArray
    } else splitHeader.map(unescapeString)

    val nField = columns.length

    val duplicates = columns.duplicates()
    if (duplicates.nonEmpty) {
      fatal(s"invalid header: found duplicate columns [${
        duplicates.map(x => '"' + x + '"').mkString(", ")
      }]")
    }


    val rdd = sc.textFilesLines(files, nPartitions)
      .filter { line =>
        commentChar.forall(ch => !line.value.startsWith(ch)) && (noHeader || line.value != header)
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
              sb.append(s"\n  Loading column `$name' as type $t (user-specified)")
              (name, t)
            case None =>
              imputedType match {
                case Some(t) =>
                  sb.append(s"\n  Loading column `$name' as type $t (imputed)")
                  (name, t)
                case None =>
                  sb.append(s"\n  Loading column `$name' as type String (no non-missing values for imputation)")
                  (name, TString)
              }
          }
        }
      } else {
        sb.append("Reading table with no type imputation\n")
        columns.map { c =>
          types.get(c) match {
            case Some(t) =>
              sb.append(s"  Loading column `$c' as type `$t' (user-specified)\n")
              (c, t)
            case None =>
              sb.append(s"  Loading column `$c' as type `String' (type not specified)\n")
              (c, TString)
          }
        }
      }
    }

    info(sb.result())

    val schema = TStruct(namesAndTypes: _*)

    val parsed = rdd
      .map {
        _.map { line =>
          val a = new Array[Annotation](nField)

          val split = splitLine(line, separator, quote)
          if (split.length != nField)
            fatal(s"expected $nField fields, but found ${ split.length } fields")

          var i = 0
          while (i < nField) {
            val (name, t) = namesAndTypes(i)
            val field = split(i)
            try {
              if (field == missing)
                a(i) = null
              else
                a(i) = TableAnnotationImpex.importAnnotation(field, t)
            } catch {
              case e: Exception =>
                fatal(s"""${ e.getClass.getName }: could not convert "$field" to $t in column "$name" """)
            }
            i += 1
          }
          Row.fromSeq(a)
        }
      }

    (schema, parsed)
  }
}
