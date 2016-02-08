package org.broadinstitute.hail.io.annotators

import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.variant.Variant


abstract class VariantAnnotator extends Serializable {

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations

  def metadata(conf: Configuration): Annotations
}

abstract class SampleAnnotator {

  def annotate(id: String, sa: Annotations): Annotations

  def metadata(): Annotations
}

object Annotator {

  def parseField(typeString: String, k: String, missing: Set[String], excluded: Set[String]):
  (String) => Option[Any] = {

    if (!excluded(k)) {
      typeString match {
        case "Double" =>
          (v: String) =>
            try {
              if (!missing(v))
                Some(v.toDouble)
              else
                None
            }
            catch {
              case e: java.lang.NumberFormatException =>
                fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Double in column "$k" """)
            }
        case "Int" =>
          (v: String) =>
            try {
              if (!missing(v))
                Some(v.toInt)
              else
                None
            }
            catch {
              case e: java.lang.NumberFormatException =>
                fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Int in column "$k" """)
            }
        case "Boolean" =>
          (v: String) =>
            try {
              if (!missing(v))
                Some(v.toBoolean)
              else
                None
            }
            catch {
              case e: java.lang.IllegalArgumentException =>
                fatal( s"""java.lang.IllegalArgumentException: tried to convert "$v" to Boolean in column "$k" """)
            }
        case "String" =>
          (v: String) =>
            if (!missing(v))
              Some(v)
            else
              None
        case _ =>
          fatal(
            s"""Unrecognized type "$typeString" in column "$k".  Hail supports the following types in annotations:
                |  - Double (floating point number)
                |  - Int  (integer)
                |  - Boolean
                |  - String
                |
               |  Note that the above types are case sensitive.
             """.stripMargin)
      }
    }
    else
      v => None
  }
}
