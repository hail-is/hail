package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{StructSignature, Signature, Annotation, SimpleSignature}
import org.broadinstitute.hail.variant.{Variant}

import scala.collection.mutable

object TSVAnnotator {
  def apply(sc: SparkContext, filename: String, vColumns: Array[String], typeMap: Map[String, String],
    missing: Set[String]): (RDD[(Variant, Annotation)], Signature) = {

    val (header, split) = readLines(filename, sc.hadoopConfiguration) { lines =>
      // fatalIf(lines.isEmpty, "empty TSV file")
      val h = lines.next().value
      (h, h.split("\t"))
    }

    val (shortForm, vColIndices) = if (vColumns.length == 1) {
      // format CHR:POS:REF:ALT
      val variantIndex = vColumns.map(s => split.indexOf(s))
      variantIndex.foreach { i =>
        fatalIf(i < 0, s"Could not find designated CHR:POS:REF:ALT column identifier '${vColumns.head}'")
      }
      (true, variantIndex)
    }
    else {
      // format CHR  POS  REF  ALT
      // lengths not equal to 1 or 4 are checked in AnnotateVariants.parseColumns
      val variantIndex = vColumns.map(s => split.indexOf(s))
      if (variantIndex(0) < 0 || variantIndex(1) < 0 || variantIndex(2) < 0 || variantIndex(2) < 0) {
        val notFound = vColumns.flatMap(i => if (split.indexOf(i) < 0) Some(i) else None)
        fatal(s"Could not find designated identifier column(s): [${notFound.mkString(", ")}]")
      }
      (false, variantIndex)
    }

    val orderedSignatures: Array[(String, Option[Signature])] = split.map { s =>
      if (!vColumns.contains(s))
        (s, Some(SimpleSignature(typeMap.getOrElse(s, "String"))))
      else
        (s, None)
    }

    val signatures = StructSignature(
      orderedSignatures.flatMap { case (key, o) => o match {
        case Some(sig) => Some(key, sig)
        case None => None
      }}
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
    (rdd, signatures)
  }
}
