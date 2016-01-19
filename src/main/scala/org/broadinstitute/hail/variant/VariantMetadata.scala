package org.broadinstitute.hail.variant

import java.io.{DataInputStream, DataOutputStream}
import java.nio.ByteBuffer
import org.apache.spark.serializer.{SerializerInstance, KryoSerializer}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._

object VariantMetadata {

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(IndexedSeq.empty[(String, String)],
    sampleIds, Annotations.emptyIndexedSeq(sampleIds.length), Annotations.empty(),
    Annotations.empty())

  def apply(filters: IndexedSeq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[Annotations], sas: Annotations, vas: Annotations): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotations],
  sampleAnnotationSignatures: Annotations,
  variantAnnotationSignatures: Annotations) {

  def nSamples: Int = sampleIds.length

  def addSampleAnnotations(sas: Annotations, sa: IndexedSeq[Annotations]): VariantMetadata =
    copy(
      sampleAnnotationSignatures = sampleAnnotationSignatures ++ sas,
      sampleAnnotations = sampleAnnotations.zip(sa).map { case (a1, a2) => a1 ++ a2 })

  def serialize(ser: SerializerInstance): WritableVariantMetadata = {
    WritableVariantMetadata(filters, sampleIds,
      ser.serialize(sampleAnnotations).array(),
      ser.serialize(sampleAnnotationSignatures).array(),
      ser.serialize(variantAnnotationSignatures).array())
  }
}

case class WritableVariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: Array[Byte],
  sampleAnnotationSignatures: Array[Byte],
  variantAnnotationSignatures: Array[Byte]) {

  def deserialize(ser: SerializerInstance): VariantMetadata = {
    VariantMetadata(filters, sampleIds,
      ser.deserialize[IndexedSeq[Annotations]](ByteBuffer.wrap(sampleAnnotations)),
      ser.deserialize[Annotations](ByteBuffer.wrap(sampleAnnotationSignatures)),
      ser.deserialize[Annotations](ByteBuffer.wrap(variantAnnotationSignatures)))
  }
}

object WritableVariantMetadata {
  implicit def writableVariantMetadata: DataWritable[WritableVariantMetadata] =
    new DataWritable[WritableVariantMetadata] {
      def write(dos: DataOutputStream, t: WritableVariantMetadata) {
        writeData(dos, t.filters)
        writeData(dos, t.sampleIds)
        writeData(dos, t.sampleAnnotations)
        writeData(dos, t.sampleAnnotationSignatures)
        writeData(dos, t.variantAnnotationSignatures)}
      }

  implicit def readableVariantMetadata: DataReadable[WritableVariantMetadata] =
    new DataReadable[WritableVariantMetadata] {
      def read(dis: DataInputStream): WritableVariantMetadata = {
        val filters = readData[IndexedSeq[(String, String)]](dis)
        val ids = readData[IndexedSeq[String]](dis)
        WritableVariantMetadata(filters, ids,
          readData[Array[Byte]](dis),
          readData[Array[Byte]](dis),
          readData[Array[Byte]](dis))
      }
    }
}