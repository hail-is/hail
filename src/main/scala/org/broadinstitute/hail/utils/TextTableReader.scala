package org.broadinstitute.hail.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.StringEscapeUtils._
import org.kohsuke.args4j.{Option => Args4jOption}

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
  val intRegex = """^-?\d+$"""
  val doubleRegex = """^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"""
  val variantRegex = """^.+:\d+:[ATGC]+:([ATGC]+|\*)(,([ATGC]+|\*))*$"""
  val locusRegex = """^.+:\d+$"""
  val headToTake = 20

  def guessType(values: Seq[String], missing: String): Option[Type] = {
    require(values.nonEmpty)

    val size = values.size

    val booleanMatch = values.exists(value => value.matches(booleanRegex))
    val variantMatch = values.exists(value => value.matches(variantRegex))
    val locusMatch = values.exists(value => value.matches(locusRegex))
    val doubleMatch = values.exists(value => value.matches(doubleRegex))
    val intMatch = values.exists(value => value.matches(intRegex))

    val defined = values.filter(_ != missing)

    val allBoolean = defined.forall(_.matches(booleanRegex))
    val allVariant = defined.forall(_.matches(variantRegex))
    val allLocus = defined.forall(_.matches(locusRegex))
    val allDouble = defined.forall(_.matches(doubleRegex))
    val allInt = defined.forall(_.matches(intRegex))

    if (values.forall(_ == missing))
      None
    else if (allBoolean && booleanMatch)
      Some(TBoolean)
    else if (allVariant && variantMatch)
      Some(TVariant)
    else if (allLocus && locusMatch)
      Some(TLocus)
    else if (allInt && intMatch)
      Some(TInt)
    else if (allDouble && doubleMatch)
      Some(TDouble)
    else
      Some(TString)
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
    val firstLines = sc.hadoopConfiguration.readLines(firstFile) { lines =>
      val filt = lines
        .filter(line => commentChar.forall(pattern => !line.value.startsWith(pattern)))

      if (filt.isEmpty)
        fatal(
          s"""invalid file: no lines remaining after comment filter
              |  Offending file: `$firstFile'
           """.stripMargin)
      else
        filt.take(headToTake).toArray
    }

    val firstLine = firstLines.head.value

    val columns = if (noHeader) {
      firstLine.split(separator, -1)
        .zipWithIndex
        .map { case (_, i) => s"_$i" }
    } else firstLine.split(separator, -1).map(unescapeString)

    val nField = columns.length

    val duplicates = columns.duplicates()
    if (duplicates.nonEmpty) {
      fatal(s"invalid header: found duplicate columns [${ duplicates.map(x => '"' + x + '"').mkString(", ") }]")
    }

    val sb = new StringBuilder

    val namesAndTypes = {
      if (impute) {
        sb.append(s"Reading table with type imputation from the leading $headToTake lines\n")
        val split = firstLines.tail.map(_.map(_.split(separator)))
        split.foreach { line =>
          line.foreach { fields =>
            if (line.value.length != nField)
              fatal(s"""$firstFile: field number mismatch: header contained $nField fields, found ${ line.value.length }""")
          }
        }

        val columnValues = Array.tabulate(nField)(i => split.map(_.value(i)))
        columns.zip(columnValues).map { case (name, col) =>
          types.get(name) match {
            case Some(t) =>
              sb.append(s"  Loading column `$name' as type $t (user-specified)\n")
              (name, t)
            case None =>
              guessType(col, missing) match {
                case Some(t) =>
                  sb.append(s"  Loading column `$name' as type $t (imputed)\n")
                  (name, t)
                case None =>
                  sb.append(s"  Loading column `$name' as type String (no non-missing values for imputation)\n")
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

    val filter: (String) => Boolean = (line: String) => {
      if (noHeader)
        true
      else line != firstLine
    } && commentChar.forall(ch => !line.startsWith(ch))

    val rdd = sc.textFilesLines(files, nPartitions)
      .filter(line => filter(line.value))
      .map {
        _.map { line =>
          val split = line.split(separator, -1)
          if (split.length != nField)
            fatal(s"expected $nField fields, but found ${ split.length } fields")
          Annotation.fromSeq(
            (split, namesAndTypes).zipped
              .map { case (elem, (name, t)) =>
                try {
                  if (elem == missing)
                    null
                  else
                    TableAnnotationImpex.importAnnotation(elem, t)
                }
                catch {
                  case e: Exception =>
                    fatal(s"""${ e.getClass.getName }: could not convert "$elem" to $t in column "$name" """)
                }
              })
        }
      }

    (schema, rdd)
  }
}
