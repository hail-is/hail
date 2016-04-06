package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

import scala.collection.mutable

object SampleTSVAnnotator extends TSVAnnotator {
  def apply(filename: String, sampleCol: String, declaredSig: Map[String, TypeWithSchema], missing: String,
    hConf: hadoop.conf.Configuration): (Map[String, Annotation], TypeWithSchema) = {
    readLines(filename, hConf) { lines =>
      fatalIf(lines.isEmpty, "empty TSV file")

      val header = lines.next().value
      val split = header.split("\t")

      val sampleIndex = split.indexOf(sampleCol)
      fatalIf(sampleIndex < 0, s"Could not find designated sample column id '$sampleCol")
      declaredSig.foreach { case (id, t) =>
        if (!split.contains(id))
          warn(s"""found "$id" in type map but not in TSV header """)
      }

      val orderedSignatures: Array[(String, Option[TypeWithSchema])] = split.map { s =>
        if (s != sampleCol) {
          val t = declaredSig.getOrElse(s, TString)
          if (!t.isInstanceOf[Parsable])
            fatal(
              s"Unsupported type $t in TSV annotation.  Supported types: Boolean, Int, Long, Float, Double and String.")
          (s, Some(t))
        } else
          (s, None)
      }

      val signature = TStruct(
        orderedSignatures.flatMap { case (key, o) =>
          o.map(sig => (key, sig))
        }: _*)

      val functions = buildParsers(missing, orderedSignatures)

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