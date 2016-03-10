package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.Variant

import scala.io.Source
import scala.collection.mutable

object SampleTSVAnnotator {
  def apply(filename: String, sampleCol: String, typeMap: Map[String, String], missing: Set[String],
    hConf: hadoop.conf.Configuration): (Map[String, Annotation], Signature) = {
    readLines(filename, hConf) { lines =>
      // fatalIf(lines.isEmpty, "empty TSV file")
      val header = lines.next().value
      val split = header.split("\t")
      val sampleIndex = split.indexOf(sampleCol)
      fatalIf(sampleIndex < 0, s"Could not find designated sample column id '$sampleCol")

      val orderedSignatures: Array[(String, Option[Signature])] = split.map { s =>
        if (s != sampleCol)
          (s, Some(SimpleSignature(typeMap.getOrElse(s, "String"))))
        else
          (s, None)
      }

      val signatures = StructSignature(
        orderedSignatures.flatMap { case (key, o) => o match {
          case Some(sig) => Some(key, sig)
          case None => None
        }
        }
          .zipWithIndex
          .map { case ((key, sig), i) => (key, (i, sig)) }
          .toMap
      )

      val functions: Array[(mutable.ArrayBuilder[Annotation], String) => Unit] =
        orderedSignatures
          .map { case (id, o) => o.map(_.parser(missing, id)) }
          .map {
            case Some(parser) =>
              (ab: mutable.ArrayBuilder[Annotation], str: String) =>
                ab += parser(str)
                ()
            case None =>
              (ab: mutable.ArrayBuilder[Annotation], str: String) => ()
          }

      val ab = mutable.ArrayBuilder.make[Any]
      val m = lines.map {
        _.transform { l =>
          val lineSplit = l.value.split("\t")
          val sample = lineSplit(sampleIndex)
          ab.clear()
          lineSplit.iterator.zip(functions.iterator)
            .foreach { case (field, fn) => fn(ab, field) }
          (sample, Row.fromSeq(ab.result()))
        }
      }
        .toMap
      (m, signatures)
    }
  }
}

//      def annotate(s: String, sa: Annotations): Annotations = {
//        sampleMap.get(s) match {
//          case Some(fields) =>
//            val sa2 = Annotations(parsedHeader.iterator.zip(fields.iterator)
//              .flatMap { case (k, v) =>
//                v match {
//                  case Some(value) => Some(k, value)
//                  case None => None
//                }
//              }
//              .toMap)
//            sa ++ rooted(sa2)
//          case None => sa
//        }
//      }
//
//      def metadata(): Annotations = signatures
//
//      def read(conf: hadoop.conf.Configuration): (IndexedSeq[String], Annotations, Map[String, Array[Option[Any]]]) = {
//        readFile[(IndexedSeq[String], Annotations, Map[String, Array[Option[Any]]])](path, conf) { reader =>
//          val lines = Source.fromInputStream(reader)
//            .getLines()
//          fatalIf(lines.isEmpty, "empty annotations file")
//
//          val header = lines
//            .next()
//            .split("\t")
//
//          val cleanHeader = header.flatMap {
//            column =>
//              if (column == sampleCol)
//                None
//              else
//                Some(column)
//          }
//
//          val keyedSignatures = header.map(column =>
//            if (column == sampleCol)
//              null
//            else
//              (column, SimpleSignature(typeMap.getOrElse(column, "String"))))
//
//          val parseFieldFunctions = keyedSignatures.map {
//            case null => null
//            case (key, sig) => sig.parser(key, missing)
//          }
//
//          val signatures = rooted(Annotations(keyedSignatures.filter(_ != null).toMap))
//
//          val sampleColIndex = header.indexOf(sampleCol)
//
//          val ab = new mutable.ArrayBuilder.ofRef[Option[Any]]
//
//          val sampleMap: Map[String, Array[Option[Any]]] = {
//            lines.map(line => {
//              val split = line.split("\t")
//              val sample = split(sampleColIndex)
//              split.iterator.zipWithIndex.foreach {
//                case (field, index) =>
//                  if (index != sampleColIndex)
//                    ab += parseFieldFunctions(index)(field)
//              }
//              val result = ab.result()
//              ab.clear()
//              (sample, ab.result())
//            })
//              .toMap
//          }
//
//          (cleanHeader, signatures, sampleMap)
//        }
//      }
//    }

