package org.broadinstitute.hail.io.annotators

import org.apache.hadoop.conf.Configuration
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.variant._

import scala.io.Source

class SampleTSVAnnotator(path: String, sampleCol: String, typeMap: Map[String, String],
  missing: Set[String], root: String) extends SampleAnnotator {

  val (parsedHeader, signatures, sampleMap) = read(new Configuration())

  val rooted = Annotator.rootFunction(root)

  def annotate(s: String, sa: Annotations): Annotations = {
    sampleMap.get(s) match {
      case Some(fields) =>
        val sa2 = Annotations(parsedHeader.iterator.zip(fields.iterator)
          .flatMap { case (k, v) =>
            v match {
              case Some(value) => Some(k, value)
              case None => None
            }
          }
          .toMap)
        sa ++ rooted(sa2)
      case None => sa
    }
  }

  def metadata(): Annotations = signatures

  def read(conf: Configuration): (IndexedSeq[String], Annotations, Map[String, IndexedSeq[Option[Any]]]) = {
    val lines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
    fatalIf(lines.isEmpty, "empty annotations file")

    val header = lines
      .next()
      .split("\t")

    val cleanHeader = header.flatMap {
      column =>
        if (column == sampleCol)
          None
        else
          Some(column)
    }

    val parseFieldFunctions = header.map(col => Annotator.parseField(
      typeMap.getOrElse(col, "String"), col, missing, Set(sampleCol)))

    val sampleColIndex = header.indexOf(sampleCol)

    val signatures = rooted(Annotations(
      cleanHeader.map(column => (column, SimpleSignature(typeMap.getOrElse(column, "String"))))
        .toMap))

    val sampleMap: Map[String, IndexedSeq[Option[Any]]] = {
      lines.map(line => {
        val split = line.split("\t")
        val sample = split(sampleColIndex)
        val indexedValues = split.iterator.zipWithIndex.map {
          case (field, index) =>
            parseFieldFunctions(index)(field)
        }
          .toIndexedSeq
        (sample, indexedValues)
      })
        .toMap
    }

    (cleanHeader, signatures, sampleMap)
  }
}

