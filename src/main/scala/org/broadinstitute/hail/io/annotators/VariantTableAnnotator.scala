package org.broadinstitute.hail.io.annotators

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Variant

import scala.collection.mutable

object VariantTableAnnotator extends TSVAnnotator {
  def apply(sc: SparkContext, files: Array[String], vColumns: Array[String], declaredSig: Map[String, Type],
    missing: String, delim: String): (RDD[(Variant, Annotation)], Type) = {

    val delimiter = unescapeString(delim)

    val (header, headerSplit) = readLines(files.head, sc.hadoopConfiguration) { lines =>
      if(lines.isEmpty)
        fatal("empty TSV file")
      val h = lines.next().value
      (h, h.split(delimiter))
    }

    val headerSet = headerSplit.toSet

    declaredSig.foreach { case (id, _) =>
      if (!headerSet(id))
        warn(s"found `$id' in type map but not in TSV header")
    }

    val (shortForm, vColIndices) = if (vColumns.length == 1) {
      // format CHR:POS:REF:ALT
      val variantIndex = vColumns.map(s => headerSplit.indexOf(s))
      variantIndex.foreach { i =>
        if(i < 0)
        fatal(s"Could not find designated CHR:POS:REF:ALT column identifier `${vColumns.head}'")
      }
      (true, variantIndex)
    } else {
      // format CHR  POS  REF  ALT
      val variantIndex = vColumns.map(s => headerSplit.indexOf(s))
      if (variantIndex(0) < 0 || variantIndex(1) < 0 || variantIndex(2) < 0 || variantIndex(2) < 0) {
        val notFound = vColumns.flatMap(i => if (headerSplit.indexOf(i) < 0) Some(i) else None)
        fatal(s"Could not find designated identifier column(s): ${notFound.mkString(", ")}")
      }
      (false, variantIndex)
    }

    if (headerSplit.length - vColIndices.length == 0)
      fatal("file contained no annotation fields")

    val orderedSignatures: Array[(String, Option[Type])] = headerSplit.map { s =>
      if (!vColumns.contains(s)) {
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

    val f: (mutable.ArrayBuilder[Annotation], Line) => (Variant, Annotation) = {
      (ab, l) =>
        l.transform { line =>
          val lineSplit = line.value.split(delimiter)
          if (lineSplit.length != headerSplit.length)
            fatal(s"Expected ${headerSplit.length} fields, but got ${lineSplit.length}")
          val variant = {
            if (shortForm) {
              // chr:pos:ref:alt
              val Array(chr, pos, ref, alt) = lineSplit(vColIndices.head).split(":")
              Variant(chr, pos.toInt, ref, alt)
            } else {
              // long form
              Variant(lineSplit(vColIndices(0)), lineSplit(vColIndices(1)).toInt,
                lineSplit(vColIndices(2)), lineSplit(vColIndices(3)))
            }
          }
          ab.clear()
          lineSplit.iterator.zip(functions.iterator)
            .foreach { case (field, fn) => fn(ab, field) }
          val res = ab.result()
          (variant, Row.fromSeq(res))
        }
    }

    val rdd = sc.textFilesLines(files)
        .filter(_.value != header)
        .mapPartitions {
          iter =>
            val ab = mutable.ArrayBuilder.make[Any]
            iter.map(line => f(ab, line))
        }

    (rdd, signature)
  }

}
