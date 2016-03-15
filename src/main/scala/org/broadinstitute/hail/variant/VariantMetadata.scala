package org.broadinstitute.hail.variant

import java.io.{DataInputStream, DataOutputStream}
import java.nio.ByteBuffer
import org.apache.spark.serializer.{SerializerInstance, KryoSerializer}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._

object VariantMetadata {

  def apply(sampleIds: Array[String]): VariantMetadata = new VariantMetadata(Array.empty[(String, String)],
    sampleIds,
    Annotation.emptyIndexedSeq(sampleIds.length),
    Signature.empty,
    Signature.empty)

  def apply(filters: IndexedSeq[(String, String)], sampleIds: Array[String],
    sa: IndexedSeq[Annotation], sas: Signature, vas: Signature): VariantMetadata = {
    new VariantMetadata(filters, sampleIds, sa, sas, vas)
  }
}

case class VariantMetadata(filters: IndexedSeq[(String, String)],
  sampleIds: IndexedSeq[String],
  sampleAnnotations: IndexedSeq[Annotation],
  saSignature: Signature,
  vaSignature: Signature,
  wasSplit: Boolean = false) {

  def nSamples: Int = sampleIds.length
}
