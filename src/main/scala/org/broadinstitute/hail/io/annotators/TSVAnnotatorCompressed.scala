package org.broadinstitute.hail.io.annotators

import java.nio.ByteBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotations, SimpleSignature}
import org.broadinstitute.hail.variant.{LZ4Utils, Variant}

import scala.io.Source

class TSVAnnotatorCompressed(path: String, vColumns: IndexedSeq[String],
  typeMap: Map[String, String], missing: Set[String], root: String)
  extends VariantAnnotator {

  @transient var internalMap: Map[Variant, (Int, Int)] = null
  @transient var compressedBlocks: IndexedSeq[(Int, Array[Byte])] = null
  @transient var headerIndex: IndexedSeq[String] = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check(sz)
    internalMap.get(v) match {
      case Some((i, j)) =>
        val (length, bytes) = compressedBlocks(i)
        val map = headerIndex.iterator.zip(
          sz.deserialize[Array[IndexedSeq[Option[Any]]]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
            .apply(j)
            .iterator)
          .flatMap {
            case (k, v) =>
              v match {
                case Some(a) => Some(k, a)
                case None => None
              }
          }
          .toMap
        va ++ rooted(Annotations(map))
      case None => va
    }
  }

  def check(sz: SerializerInstance) {
    if (internalMap == null)
      read(new Configuration(), sz)
  }

  def metadata(conf: Configuration): Annotations = {
    readFile(path, conf)(is => {
      val header = Source.fromInputStream(is)
        .getLines()
        .next()
        .split("\t")

      val (chrIndex, posIndex, refIndex, altIndex) =
        (header.indexOf(vColumns(0)),
          header.indexOf(vColumns(1)),
          header.indexOf(vColumns(2)),
          header.indexOf(vColumns(3)))

      if (chrIndex < 0 || posIndex < 0 || refIndex < 0 || altIndex < 0) {
        val notFound = vColumns.flatMap(i => if (header.indexOf(i) < 0) Some(i) else None)
        fatal(s"Could not find designated identifier column(s): [${notFound.mkString(", ")}]")
      }

      val anno = Annotations(header.flatMap(s =>
        if (!vColumns.toSet(s))
          Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
        else
          None)
        .toMap)

      rooted(anno)
    })
  }

  def read(conf: Configuration, serializer: SerializerInstance) {
    val lines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
      .filter(line => !line.isEmpty)

    val header = lines.next()
      .split("\t")
    val (chrIndex, posIndex, refIndex, altIndex) =
      (header.indexOf(vColumns(0)),
        header.indexOf(vColumns(1)),
        header.indexOf(vColumns(2)),
        header.indexOf(vColumns(3)))

    headerIndex = header.flatMap(line => if (!vColumns.contains(line)) Some(line) else None).toIndexedSeq

    if (chrIndex < 0 || posIndex < 0 || refIndex < 0 || altIndex < 0) {
      val notFound = vColumns.flatMap(i => if (header.indexOf(i) < 0) Some(i) else None)
      fatal(s"Could not find designated identifier column(s): [${notFound.mkString(", ")}]")
    }

    val signatures = Annotations(header.flatMap(s =>
      if (!vColumns.toSet(s))
        Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
      else
        None)
      .toMap)

    val excluded = vColumns.toSet
    val functions = header.map(col => Annotator.parseField(typeMap.getOrElse(col, "String"), col, missing, excluded)).toIndexedSeq
    val bbb = new ByteBlockBuilder[IndexedSeq[Option[Any]]](serializer)

    var i = 1
    val variantMap = lines.map {
      line =>
        i += 1
        val split = line.split("\t")
        val indexedValues = split.iterator.zipWithIndex.flatMap {
          case (field, index) =>
            if (index != chrIndex && index != posIndex && index != refIndex && index != altIndex)
              Some(functions(index)(field))
            else
              None
        }
          .toIndexedSeq
        if (i % 10000 == 0)
          println("line " + i + " " + indexedValues)
        val variantIndex = bbb.add(indexedValues)
        (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), variantIndex)
    }
      .toMap

    internalMap = variantMap
    compressedBlocks = bbb.result()
  }

  def serialize(path: String, sz: SerializerInstance) {
    val conf = new Configuration()
    val signatures = metadata(conf)
    check(sz)

    val stream = sz.serializeStream(hadoopCreate(path, conf))
      .writeObject[String]("tsv")
      .writeObject[IndexedSeq[String]](headerIndex)
      .writeObject[Annotations](signatures)
      .writeObject(internalMap)
      .writeObject(compressedBlocks)
      .close()
  }
}
