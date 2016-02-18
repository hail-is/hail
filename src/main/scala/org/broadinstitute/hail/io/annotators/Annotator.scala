package org.broadinstitute.hail.io.annotators

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.variant.Variant

class SerializableHadoopConfiguration(@transient var value: Configuration) extends Serializable {
  private def writeObject(out: ObjectOutputStream) {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream) {
    value = new Configuration(false)
    value.readFields(in)
  }
}

abstract class VariantAnnotator extends Serializable {

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations

  def metadata(): Annotations
}

abstract class SampleAnnotator {

  def annotate(id: String, sa: Annotations): Annotations

  def metadata(): Annotations
}

object Annotator {

  def rootFunction(root: String): Annotations => Annotations = {
    root match {
      case null =>
        va => va
      case r =>
        va => r
          .split("""\.""")
          .filter(!_.isEmpty)
          .foldRight(va)((id, annotations) => Annotations(Map(id -> annotations)))
    }
  }

  def parseField(typeString: String, k: String,
    missing: Set[String]): (String) => Option[Any] = {

    typeString match {
      case "Double" =>
        (v: String) =>
          try {
            someIf(!missing(v), v.toDouble)
          } catch {
            case e: java.lang.NumberFormatException =>
              fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Double in column "$k" """)
          }
      case "Int" =>
        (v: String) =>
          try {
            someIf(!missing(v), v.toInt)
          } catch {
            case e: java.lang.NumberFormatException =>
              fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Int in column "$k" """)
          }
      case "Boolean" =>
        (v: String) =>
          try {
            someIf(!missing(v), v.toBoolean)
          } catch {
            case e: java.lang.IllegalArgumentException =>
              fatal( s"""java.lang.IllegalArgumentException: tried to convert "$v" to Boolean in column "$k" """)
          }
      case "String" =>
        (v: String) =>
          someIf(!missing(v), v)
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
}
