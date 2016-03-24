package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr

import scala.collection.mutable

object SampleTSVAnnotator {
  def apply(filename: String, sampleCol: String, typeMap: Map[String, String], missing: Set[String],
    hConf: hadoop.conf.Configuration): (Map[String, Annotation], expr.Type) = {
    readLines(filename, hConf) { lines =>
      fatalIf(lines.isEmpty, "empty TSV file")

      val header = lines.next().value
      val split = header.split("\t")

      val sampleIndex = split.indexOf(sampleCol)
      fatalIf(sampleIndex < 0, s"Could not find designated sample column id '$sampleCol")
      typeMap.foreach { case (id, t) =>
        if (!split.contains(id))
          warn(s"""found "$id" in type map but not in TSV header """)
      }

      val orderedSignatures: Array[(String, Option[expr.Type])] = split.map { s =>
        if (s != sampleCol)
          (s, Some(VariantTSVAnnotator.parseStringType(typeMap.getOrElse(s, "String"))))
        else
          (s, None)
      }

      val signature = expr.TStruct(
        orderedSignatures.flatMap { case (key, o) =>
          o match {
            case Some(sig) => Some(key, sig)
            case None => None
          }
        }
          .zipWithIndex
          .map { case ((key, t), i) => expr.Field(key, t, i) }
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
      (m, signature)
    }
  }
}