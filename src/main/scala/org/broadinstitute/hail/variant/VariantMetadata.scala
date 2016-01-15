package org.broadinstitute.hail.variant

import java.io.{DataInputStream, DataOutputStream}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._

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
      readData[IndexedSeq[Annotations]](dis),
      readData[Annotations](dis),
      readData[Annotations](dis))
    }
  }

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
}
