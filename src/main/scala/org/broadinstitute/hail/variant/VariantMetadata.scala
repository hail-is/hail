package org.broadinstitute.hail.variant

import java.io.{DataInputStream, DataOutputStream}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.annotations.AnnotationSiganture._

object VariantMetadata {
  implicit def writableVariantMetadata: DataWritable[VariantMetadata] =
    new DataWritable[VariantMetadata] {
      def write(dos: DataOutputStream, t: VariantMetadata) {
        writeData(dos, t.filters)
        writeData(dos, t.sampleIds)
        writeData(dos, t.sampleAnnotations)
        writeData(dos, t.sampleAnnotationSignatures)
        writeData(dos, t.variantAnnotationSignatures)
      }
    }

  implicit def readableVariantMetadata: DataReadable[VariantMetadata] =
  new DataReadable[VariantMetadata] {
    def read(dis: DataInputStream): VariantMetadata = {
      VariantMetadata(readData[IndexedSeq[(String, String)]](dis),
      readData[IndexedSeq[String]](dis),
      readData[IndexedSeq[AnnotationData]](dis),
      readData[AnnotationSignatures](dis),
      readData[AnnotationSignatures](dis))
    }
  }

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(IndexedSeq.empty[(String, String)],
    sampleIds, Annotations.emptyOfArrayString(sampleIds.length), Annotations.emptyOfSignature(),
    Annotations.emptyOfSignature())

  def apply(filters: IndexedSeq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[AnnotationData], sas: AnnotationSignatures, vas: AnnotationSignatures): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[AnnotationData],
  sampleAnnotationSignatures: AnnotationSignatures,
  variantAnnotationSignatures: AnnotationSignatures) {

  def nSamples: Int = sampleIds.length

  def addSampleAnnotations(sas: AnnotationSignatures, sa: IndexedSeq[AnnotationData]): VariantMetadata =
    copy(
      sampleAnnotationSignatures = sampleAnnotationSignatures ++ sas,
      sampleAnnotations = sampleAnnotations.zip(sa).map { case (a1, a2) => a1 ++ a2 })
}
