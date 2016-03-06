package org.broadinstitute.hail.variant

import java.io.{DataInputStream, DataOutputStream}
import java.nio.ByteBuffer
import org.apache.spark.serializer.{SerializerInstance, KryoSerializer}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._

object VariantMetadata {

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(Array.empty[(String, String)],
    sampleIds,
    Annotations.emptyIndexedSeq(sampleIds.length),
    Annotations.emptySignature,
    Annotations.emptySignature)

  def apply(filters: IndexedSeq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[Annotation], sas: StructSignature, vas: StructSignature): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotation],
  sampleAnnotationSignatures: Signature,
  variantAnnotationSignatures: Signature,
  wasSplit: Boolean = false) {

  def nSamples: Int = sampleIds.length

//  def addSampleAnnotations(sas: StructSignature, sa: IndexedSeq[AnnotationData]): VariantMetadata =
//    copy(
//      sampleAnnotationSignatures = sampleAnnotationSignatures ++ sas,
//      sampleAnnotations = sampleAnnotations.zip(sa).map { case (a1, a2) => a1 ++ a2 })

//  def addVariantAnnotationSignatures(newSigs: StructSignature): VariantMetadata =
//    copy(
//      variantAnnotationSignatures = variantAnnotationSignatures ++ newSigs)
//
//  def addVariantAnnotationSignatures(key: String, sig: Any): VariantMetadata =
//    copy(
//      variantAnnotationSignatures = variantAnnotationSignatures +(key, sig))
//
//  def removeVariantAnnotationSignatures(key: String): VariantMetadata =
//    copy(
//      variantAnnotationSignatures = variantAnnotationSignatures - key)
}
