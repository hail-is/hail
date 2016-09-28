package org.broadinstitute.hail.utils

import java.util.regex.Pattern

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.StringEscapeUtils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

trait TextTableOptions {
  @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
    usage = "Define types of fields in annotations files")
  var types: String = _

  @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
    usage = "Specify identifier to be treated as missing")
  var missingIdentifier: String = "NA"

  @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
    usage = "Field delimiter regex")
  var separator: String = "\\t"

  @Args4jOption(required = false, name = "--comment",
    usage = "Skip lines beginning with the given pattern")
  var commentChar: String = _

  @Args4jOption(required = false, name = "--no-header",
    usage = "indicate that the file has no header and columns should be indicated by `_1, _2, ... _N' (0-indexed)")
  var noHeader: Boolean = _

  @Args4jOption(required = false, name = "--impute",
    usage = "impute column types")
  var impute: Boolean = _

  def config: TextTableConfiguration = TextTableConfiguration(
    types = Parser.parseAnnotationTypes(Option(types).getOrElse("")),
    noHeader = noHeader,
    impute = impute,
    separator = separator,
    missing = missingIdentifier,
    commentChar = Option(commentChar)
  )
}

case class TextTableConfiguration(
  types: Map[String, Type] = Map.empty[String, Type],
  commentChar: Option[String] = None,
  separator: String = "\t",
  missing: String = "NA",
  noHeader: Boolean = false,
  impute: Boolean = false)

object TextTableReader {

  val booleanRegex = """^([Tt]rue)|([Ff]alse)|(TRUE)|(FALSE)$"""
  val variantRegex = """^.+:\d+:[ATGC]+:([ATGC]+|\*)(,([ATGC]+|\*))*$"""
  val locusRegex = """^.+:\d+$"""
  val doubleRegex = """^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"""
  val intRegex = """^-?\d+$"""

  def imputeTypes(values: RDD[WithContext[String]], header: Array[String],
    delimiter: String, missing: String): Array[Option[Type]] = {
    val nFields = header.length
    val regexes = Array(booleanRegex, variantRegex, locusRegex, intRegex, doubleRegex).map(Pattern.compile)

    val regexTypes: Array[Type] = Array(TBoolean, TVariant, TLocus, TInt, TDouble)

    val nRegex = regexes.length
    val missingIndex = 2 * nRegex

    def zero() = {
      val m = MultiArray2.fill[Boolean](nFields, missingIndex + 1)(true)
      for (i <- 0 until nFields)
        for (j <- 0 until nRegex)
          m.update(i, j, false)
      m
    }

    val imputation = values.treeAggregate(zero())({ case (ma, line) =>
      line.foreach { l =>
        val split = l.split(delimiter)
        if (split.length != nFields)
          fatal(s"expected $nFields fields, but found ${ split.length }")

        var i = 0
        while (i < nFields) {
          val field = split(i)
          if (field != missing) {
            var j = 0
            while (j < nRegex) {
              val matches = regexes(j).matcher(field).matches()
              ma.update(i, j, ma(i, j) || matches)
              ma.update(i, j + nRegex, ma(i, j + nRegex) && (field == missing || matches))
              j += 1
            }
            ma.update(i, missingIndex, false)
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
          ma1.update(i, j, ma1(i, j) || ma2(i, j))
          ma1.update(i, j + nRegex, ma1(i, j + nRegex) && ma2(i, j + nRegex))
          j += 1
        }
        ma1.update(i, missingIndex, ma1(i, missingIndex) && ma2(i, missingIndex))
        i += 1
      }
      ma1
    })

    Array.tabulate(nFields) { i =>
      if (imputation(i, missingIndex))
        None
      else {
        Some((0 until nRegex).find { j =>
          imputation(i, j) && imputation(i, j + nRegex)
        }.map(regexTypes)
          .getOrElse(TString))
      }
    }
  }

  def read(sc: SparkContext)(files: Array[String],
    config: TextTableConfiguration = TextTableConfiguration(),
    nPartitions: Int = sc.defaultMinPartitions): (TStruct, RDD[WithContext[Annotation]]) = {
    require(files.nonEmpty)

    val noHeader = config.noHeader
    val impute = config.impute
    val separator = config.separator
    val commentChar = config.commentChar
    val missing = config.missing
    val types = config.types

    val firstFile = files.head
    val header = sc.hadoopConfiguration.readLines(firstFile) { lines =>
      val filt = lines
        .filter(line => commentChar.forall(pattern => !line.value.startsWith(pattern)))

      if (filt.isEmpty)
        fatal(
          s"""invalid file: no lines remaining after comment filter
              |  Offending file: `$firstFile'
           """.stripMargin)
      else
        filt.next().value
    }

    val columns = if (noHeader) {
      header.split(separator, -1)
        .zipWithIndex
        .map {
          case (_, i) => s"_$i"
        }
    } else header.split(separator, -1).map(unescapeString)

    val nField = columns.length

    val duplicates = columns.duplicates()
    if (duplicates.nonEmpty) {
      fatal(s"invalid header: found duplicate columns [${
        duplicates.map(x => '"' + x + '"').mkString(", ")
      }]")
    }


    val rdd = sc.textFilesLines(files, nPartitions)
      .filter { line =>
        commentChar.forall(ch => !line.value.startsWith(ch)) && {
          if (noHeader)
            true
          else
            line.value != header
        }
      }

    val sb = new StringBuilder

    val namesAndTypes = {
      if (impute) {
        info("Reading table to impute column types")

        sb.append("Finished type imputation")
        val imputedTypes = imputeTypes(rdd, columns, separator, missing)
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

    val ab = mutable.ArrayBuilder.make[Annotation]
    val parsed = rdd
      .map {
        _.map { line =>
          val split = line.split(separator, -1)
          if (split.length != nField)
            fatal(s"expected $nField fields, but found ${ split.length } fields")

          ab.clear()
          var i = 0
          while (i < nField) {
            val (name, t) = namesAndTypes(i)
            val field = split(i)
            try {
              if (field == missing)
                ab += null
              else
                ab += TableAnnotationImpex.importAnnotation(field, t)
            } catch {
              case e: Exception =>
                fatal(s"""${ e.getClass.getName }: could not convert "$field" to $t in column "$name" """)
            }
            i += 1
          }
          val a = Annotation.fromSeq(ab.result())
          a
        }
      }

    (schema, parsed)
  }
}
