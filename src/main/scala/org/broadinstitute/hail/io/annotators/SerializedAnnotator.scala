package org.broadinstitute.hail.io.annotators

import java.nio.ByteBuffer

import org.apache.hadoop
import org.apache.spark.SparkEnv
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.variant.{LZ4Utils, Variant}

class SerializedAnnotator(path: String, root: String, hConf: hadoop.conf.Configuration) extends VariantAnnotator {

  val conf = new SerializableHadoopConfiguration(hConf)
  @transient var variantIndexes: Map[Variant, (Int, Int)] = null
  @transient var compressedBytes: IndexedSeq[(Int, Array[Byte])] = null
  @transient var cleanHeader: Array[String] = null
  @transient var f: (Variant, Annotations, SerializerInstance) => Annotations = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check(sz)
    f(v, va, sz)
  }

  def check(sz: SerializerInstance) {
    if (variantIndexes == null || compressedBytes == null)
      read(sz)
  }

  def read(sz: SerializerInstance) {
    readFile(path, conf.value) { reader =>
      val dsStream = sz
        .deserializeStream(reader)

      val inputType = dsStream.readObject[String]
      cleanHeader = dsStream.readObject[Array[String]]
      val sigs = dsStream.readObject[Annotations]
      variantIndexes = {
        val length = dsStream.readObject[Int]
        (0 until length).map(i => dsStream.readObject[(Variant, (Int, Int))])
          .toMap
      }
      compressedBytes = {
        val length = dsStream.readObject[Int]
        (0 until length).map(i => dsStream.readObject[(Int, Array[Byte])])
      }

      f = {
        inputType match {
          case "tsv" =>
            (v, va, sz) => variantIndexes.get(v) match {
              case Some((i, j)) =>
                val (length, bytes) = compressedBytes(i)
                va ++ rooted(Annotations(cleanHeader.iterator.zip(sz.deserialize[Array[Array[Option[Any]]]](
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
  }

  def metadata(): Annotations = {
    readFile(path, conf.value) { reader =>
      val serializerStream = SparkEnv.get
        .serializer
        .newInstance()
        .deserializeStream(reader)
      val (source, header) = (serializerStream.readObject[String],
        serializerStream.readObject[Array[String]])
      rooted(serializerStream.readObject[Annotations])
    }
  }
}