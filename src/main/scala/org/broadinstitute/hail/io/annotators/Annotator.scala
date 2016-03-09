package org.broadinstitute.hail.io.annotators

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.Variant

class SerializableHadoopConfiguration(@transient var value: hadoop.conf.Configuration) extends Serializable {
  private def writeObject(out: ObjectOutputStream) {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream) {
    value = new hadoop.conf.Configuration(false)
    value.readFields(in)
  }
}

abstract class VariantAnnotator[T] extends Serializable {

  def rdd: RDD[(Variant, Annotation)]
  def annotate(v: Variant, va: Annotation, sz: SerializerInstance): Annotation

  def metadata(): Signature
}

abstract class SampleAnnotator {

  def annotate(id: String, sa: Annotation): Annotation

  def metadata(): Signature
}

object Annotator {

  def rootFunction(root: String): Annotation => Annotation = {
    root match {
      case null =>
        va => va
      case r =>
        val split = r.split("""\.""")
        fatalIf(!split.forall(_.length > 0), s"Invalid input: found an empty identifier in '$root'")
        va =>
          split.foldRight(va)((id, annotations) => Annotation(Map(id -> annotations)))
    }
  }
}
