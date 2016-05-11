package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

import scala.collection.mutable

object GlobalTableAnnotator extends TSVAnnotator {
  def apply(filename: String, hConf: hadoop.conf.Configuration, declaredSig: Map[String, Type], missing: String,
    delim: String): (IndexedSeq[Annotation], Type) = {
    readLines(filename, hConf) { lines =>
      if (lines.isEmpty)
        fatal("empty file")

      val delimiter = unescapeString(delim)

      val header = lines.next().value
      val split = header.split(delimiter)

      val orderedSignatures: Array[(String, Option[Type])] = split.map { s =>
        val t = declaredSig.getOrElse(s, TString)
        if (!t.isInstanceOf[Parsable])
          fatal(
            s"Unsupported type $t in TSV annotation.  Supported types: Boolean, Int, Long, Float, Double and String.")
        (s, Some(t))
      }

      val signature = TArray(TStruct(
        orderedSignatures.flatMap { case (key, o) =>
          o.map(sig => (key, sig))
        }: _*))

      val functions = buildParsers(missing, orderedSignatures)

      val ab = mutable.ArrayBuilder.make[Any]
      val m = lines.map {
        _.transform { l =>
          val lineSplit = l.value.split(delimiter)
          if (lineSplit.length != split.length)
            fatal(s"expected ${split.length} fields, but got ${lineSplit.length}")
          ab.clear()
          lineSplit.iterator.zip(functions.iterator)
            .foreach { case (field, fn) => fn(ab, field) }
          Annotation.fromSeq(ab.result())
        }
      }
        .toIndexedSeq
      (m, signature)
    }
  }
}