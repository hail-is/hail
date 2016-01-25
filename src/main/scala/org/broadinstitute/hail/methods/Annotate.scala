package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.variant.{RichVDS, VariantDataset}
import org.broadinstitute.hail.Utils._

import scala.io.Source

object Annotate {

  def convertType(typeString: String, column: String, missing: Set[String])(a: String): Option[Any] = {
    try {
      if (missing(a))
        None
      else {
        typeString match {
          case "Double" => Some(a.toDouble)
          case "Float" => Some(a.toDouble)
          case "Int" => Some(a.toInt)
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

  def annotateSamplesFromTSV(vds: VariantDataset, path: String, root: String, sampleCol: String,
    typeMap: Map[String, String], missing: Set[String]): VariantDataset = {
    val lines = Source.fromInputStream(hadoopOpen(path, vds.sparkContext.hadoopConfiguration))
      .getLines()

    if (lines.isEmpty)
      fatal("empty annotations file")

    val header = lines
      .next()
      .split("\t")

    val functions = header.map(col => convertType(typeMap.getOrElse(col, "String"), col, missing)(_))

    val sampleColIndex = header.indexOf(sampleCol)

    val sampleMap: Map[String, Annotations] = {
      val fields = lines.map(line => {
        val split = line.split("\t")
        val sample = split(sampleColIndex)
        (sample, Annotations(split.zipWithIndex
          .flatMap {
            case (field, index) =>
              functions(index)(field) match {
                case Some(ret) => Some(header(index), ret)
                case None => None
              }
          }
          .toMap - sampleCol))
      })
        .toMap
      if (root == null)
        fields
      else
        fields.mapValues(v => Annotations(Map(root -> v)))
    }

    println(sampleMap)

    val newSampleAnnotations = vds.sampleIds
      .map(id => sampleMap.getOrElse(id, Annotations.empty()))

    val signatures = header.flatMap { col =>
      if (col == sampleCol)
        None
      else
        Some(col, SimpleSignature(typeMap.getOrElse(col, "String")))
    }
      .toMap

    val sigsToAdd = {
      if (root == null)
        Annotations(signatures)
      else
        Annotations(Map(root -> Annotations(signatures)))
    }
    println(sigsToAdd)

    val localIds = vds.localSamples.map(vds.sampleIds)
    val overlap = localIds.map(sampleMap.contains)
    val missingSamples = overlap.count(_ == false)
    if (missingSamples > 0)
      println(s"WARNING: $missing local samples not found in annotations file")

    vds.copy(metadata = vds.metadata.addSampleAnnotations(
      sigsToAdd, newSampleAnnotations))
  }

  def annotateVariantsFromTSV(vds: VariantDataset, path: String, root: String, missing: Set[String]) : VariantDataset = {
    throw new UnsupportedOperationException
  }
}
