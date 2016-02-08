package org.broadinstitute.hail.io.annotators

import java.nio.ByteBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.variant.{LZ4Utils, Variant}

class SerializedAnnotator(path: String, root: String) extends VariantAnnotator {
  @transient var variantIndexes: Map[Variant, (Int, Int)] = null
  @transient var compressedBytes: IndexedSeq[(Int, Array[Byte])] = null
  @transient var cleanHeader: IndexedSeq[String] = null
  @transient var f: (Variant, Annotations, SerializerInstance) => Annotations = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check(sz)
    f(v, va, sz)
  }

  def check(sz: SerializerInstance) {
    if (variantIndexes == null || compressedBytes == null)
      read(new Configuration, sz)
  }

  def read(conf: Configuration, sz: SerializerInstance) {
    val t0 = System.nanoTime()
    val dsStream = sz
      .deserializeStream(hadoopOpen(path, conf))

    val inputType = dsStream.readObject[String]
    cleanHeader = dsStream.readObject[IndexedSeq[String]]
    val sigs = dsStream.readObject[Annotations]
    variantIndexes = dsStream.readObject[Map[Variant, (Int, Int)]]
    compressedBytes = dsStream.readObject[IndexedSeq[(Int, Array[Byte])]]

    f = {
      inputType match {
        case "tsv" =>
          (v, va, sz) => variantIndexes.get(v) match {
            case Some((i, j)) =>
              val (length, bytes) = compressedBytes(i)
              va ++ rooted(Annotations(cleanHeader.iterator.zip(sz.deserialize[Array[IndexedSeq[Option[Any]]]](
                ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
                .apply(j)
                .iterator)
                .flatMap {
                  case (k, v) => v match {
                    case Some(value) => Some(k, value)
                    case None => None
                  }
                }
                .toMap))
            case None => va
          }
        case "vcf" =>
          (v, va, sz) => variantIndexes.get(v) match {
            case Some((i, j)) =>
              val (length, bytes) = compressedBytes(i)
              va ++ rooted(sz.deserialize[Array[Annotations]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
                .apply(j))
            case None => va
          }
        case _ => throw new UnsupportedOperationException
      }
    }
  }

  def metadata(conf: Configuration): Annotations = {
    val serializerStream = SparkEnv.get
      .serializer
      .newInstance()
      .deserializeStream(hadoopOpen(path, conf))
    val (source, header) = (serializerStream.readObject[String],
      serializerStream.readObject[IndexedSeq[String]])
    rooted(serializerStream.readObject[Annotations])
  }
}