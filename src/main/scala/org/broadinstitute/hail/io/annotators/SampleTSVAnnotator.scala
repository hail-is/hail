package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}

import scala.io.Source
import scala.collection.mutable

class SampleTSVAnnotator(path: String, sampleCol: String, typeMap: Map[String, String],
  missing: Set[String], root: String, conf: hadoop.conf.Configuration) extends SampleAnnotator {

  val rooted = Annotator.rootFunction(root)
  val (parsedHeader, signatures, sampleMap) = read(conf)


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

  def read(conf: hadoop.conf.Configuration): (IndexedSeq[String], Annotations, Map[String, Array[Option[Any]]]) = {
    readFile[(IndexedSeq[String], Annotations, Map[String, Array[Option[Any]]])](path, conf) { reader =>
      val lines = Source.fromInputStream(reader)
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

      val keyedSignatures = header.map(column =>
        if (column == sampleCol)
          null
        else
        (column, SimpleSignature(typeMap.getOrElse(column, "String"))))

      val parseFieldFunctions = keyedSignatures.map {
        case null => null
        case (key, sig) => sig.parser(key, missing)}

      val signatures = rooted(Annotations(keyedSignatures.filter(_ != null).toMap))

      val sampleColIndex = header.indexOf(sampleCol)

      val ab = new mutable.ArrayBuilder.ofRef[Option[Any]]

      val sampleMap: Map[String, Array[Option[Any]]] = {
        lines.map(line => {
          val split = line.split("\t")
          val sample = split(sampleColIndex)
          split.iterator.zipWithIndex.foreach {
            case (field, index) =>
              if (index != sampleColIndex)
                ab += parseFieldFunctions(index)(field)
          }
          val result = ab.result()
          ab.clear()
          (sample, ab.result())
        })
          .toMap
      }

      (cleanHeader, signatures, sampleMap)
    }
  }
}

