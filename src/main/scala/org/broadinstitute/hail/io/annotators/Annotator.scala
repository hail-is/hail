package org.broadinstitute.hail.io.annotators

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
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
        val split = r.split("""\.""")
        fatalIf(!split.forall(_.length > 0), s"Invalid input: found an empty identifier in '$root'")
        va =>
          split.foldRight(va)((id, annotations) => Annotations(Map(id -> annotations)))
    }
  }
}
