package org.broadinstitute.hail.io.annotators

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Variant

import scala.collection.mutable

object VariantTSVAnnotator extends TSVAnnotator {
  def apply(sc: SparkContext, filename: String, vColumns: Array[String], declaredSig: Map[String, Type],
    missing: String): (RDD[(Variant, Annotation)], Type) = {

    val (header, split) = readLines(filename, sc.hadoopConfiguration) { lines =>
      fatalIf(lines.isEmpty, "empty TSV file")
      val h = lines.next().value
      (h, h.split("\t"))
    }

    declaredSig.foreach { case (id, _) =>
      if (!split.contains(id))
        warn(s"found `$id' in type map but not in TSV header")
    }

    val (shortForm, vColIndices) = if (vColumns.length == 1) {
      // format CHR:POS:REF:ALT
      val variantIndex = vColumns.map(s => split.indexOf(s))
      variantIndex.foreach { i =>
        fatalIf(i < 0, s"Could not find designated CHR:POS:REF:ALT column identifier `${vColumns.head}'")
      }
      (true, variantIndex)
    } else {
      // format CHR  POS  REF  ALT
      // lengths not equal to 1 or 4 are checked in AnnotateVariants.parseColumns
      val variantIndex = vColumns.map(s => split.indexOf(s))
      if (variantIndex(0) < 0 || variantIndex(1) < 0 || variantIndex(2) < 0 || variantIndex(2) < 0) {
        val notFound = vColumns.flatMap(i => if (split.indexOf(i) < 0) Some(i) else None)
        fatal(s"Could not find designated identifier column(s): ${notFound.mkString(", ")}")
      }
      (false, variantIndex)
    }

    val orderedSignatures: Array[(String, Option[Type])] = split.map { s =>
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

    val f: (mutable.ArrayBuilder[Annotation], String) => (Variant, Annotation) = {
      (ab, line) =>
        val lineSplit = line.split("\t")
        val variant = {
          if (shortForm) {
            // chr:pos:ref:alt
            val Array(chr, pos, ref, alt) = lineSplit(vColIndices.head).split(":")
            Variant(chr, pos.toInt, ref, alt)
          }
          else {
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

    val rdd = sc.textFile(filename)
      .filter(line => line != header)
      .mapPartitions {
        iter =>
          val ab = mutable.ArrayBuilder.make[Any]
          iter.map(line => f(ab, line))
      }
    (rdd, signature)
  }

}
