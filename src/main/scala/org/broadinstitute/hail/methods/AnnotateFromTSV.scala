package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.variant.VariantDataset
import org.broadinstitute.hail.Utils._

import scala.io.Source

object AnnotateSamples {

  def convertType(typeString: String, column: String)(a: String): Option[Any] = {
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

  def fromTSV(vds: VariantDataset, path: String, name: String, typeMap: Map[String, String], sampleCol: String): VariantDataset = {
    val lines = Source.fromInputStream(hadoopOpen(path, vds.sparkContext.hadoopConfiguration))
      .getLines()

    if (lines.isEmpty)
      fatal("empty annotations file")

    val header = lines
      .next()
      .split("\t")

    val functions = header.map(col => convertType(typeMap.getOrElse(col, "String"), col))

    val sampleColIndex = header.indexOf(sampleCol)

    val sampleMap: Map[String, Map[String, Any]] =
      lines.map(line => {
        val split = line.split("\t")
        val sample = split(sampleColIndex)
        (sample, split.zipWithIndex
          .flatMap {
            case (field, index) =>
              functions(index)(field) match {
                case Some(ret) => Some(field, ret)
                case None => None
              }
          }
          .toMap - sampleCol)
      })
      .toMap

    val signatures = typeMap.mapValues(i => SimpleSignature(i, s"to$i"))


    }
  }
