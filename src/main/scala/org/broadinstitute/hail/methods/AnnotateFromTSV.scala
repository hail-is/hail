package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant.VariantDataset
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.CmdLineException

import scala.io.Source

object AnnotateSamples {

  def convertType(a: String, typeString: String, column: String): Option[Any] = {
    try {

      a match {
        case "NA" => None
        case _ => typeString match {
          case "Double" => Some(a.toDouble)
          case "Int" => Some(a.toDouble)
          case "Boolean" => Some(a.toBoolean)
          case _ => Some(a)
        }
      }
    }
    catch {
      case e: java.lang.NumberFormatException =>
        println( s"""java.lang.NumberFormatException: tried to convert "$a" to $typeString in column "$column" """)
        sys.exit(1)
    }

  }

  def fromTSV(path: String, vds: VariantDataset): VariantDataset = {
    val lines = Source.fromInputStream(hadoopOpen(path, vds.sparkContext.hadoopConfiguration))
      .getLines()

    if (lines.isEmpty)
      fatal("empty annotations file")

    val header = lines
      .next()
      .split("\t")

    if (!(header(0).toLowerCase == "sample"))
      fatal("first column of annotations file must be 'Sample'")

    val namesAndTypes = header
      .takeRight(header.length - 1)
      .map(_.split(":").map(_.trim))
      .map(arr =>
        if (arr.length == 1)
          Array(arr(0), "String")
        else
          arr)

    lines
      .map(_.split("\t"))
      .map(i => (i(0), i.takeRight(i.length - 1)))
      .toMap
      .mapValues(arr =>
       namesAndTypes.zip(arr))

  }

}
