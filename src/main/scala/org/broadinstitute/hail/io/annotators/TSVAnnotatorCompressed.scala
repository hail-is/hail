package org.broadinstitute.hail.io.annotators

import java.nio.ByteBuffer

import org.apache.hadoop
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotations, SimpleSignature}
import org.broadinstitute.hail.variant.{Variant, LZ4Utils}

import scala.io.Source
import scala.collection.mutable

class TSVAnnotatorCompressed(path: String, vColumns: IndexedSeq[String],
  typeMap: Map[String, String], missing: Set[String], root: String)
  extends VariantAnnotator {

  @transient var internalMap: Map[Variant, (Int, Int)] = null
  @transient var compressedBlocks: Array[(Int, Array[Byte])] = null
  @transient var headerIndex: Array[String] = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check(sz)
    internalMap.get(v) match {
      case Some((i, j)) =>
        val (length, bytes) = compressedBlocks(i)
        val map = headerIndex.iterator.zip(
          sz.deserialize[Array[Array[Option[Any]]]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
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
      read(new hadoop.conf.Configuration(), sz)
  }

  def metadata(conf: hadoop.conf.Configuration): Annotations = {
    readFile(path, conf)(is => {
      val header = Source.fromInputStream(is)
        .getLines()
        .next()
        .split("\t")

      val anno = Annotations(header.flatMap(s =>
        if (!vColumns.contains(s))
          Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
        else
          None)
        .toMap)

      rooted(anno)
    })
  }

  def read(conf: hadoop.conf.Configuration, serializer: SerializerInstance) {
    readFile(path, conf) {
      reader =>
        val lines = Source.fromInputStream(reader)
          .getLines()
          .filter(line => !line.isEmpty)


        val header = lines.next()
          .split("\t")

        headerIndex = header.flatMap { col =>
          if (!vColumns.contains(col))
            Some(col)
          else
            None
        }

        val excluded = vColumns.toSet
        val functions = header.map(col => Annotator.parseField(typeMap.getOrElse(col, "String"), col, missing))
        val bbb = new ByteBlockBuilder[Array[Option[Any]]](serializer)
        val ab = new mutable.ArrayBuilder.ofRef[Option[Any]]

        val parseLine: String => (Variant, Array[Option[Any]]) = {
          // format CHR:POS:REF:ALT
          if (vColumns.length == 1) {
            val variantIndex = header.indexOf(vColumns.head)
            fatalIf(variantIndex < 0, s"Could not find designated CHR:POS:REF:ALT column identifier '${vColumns.head}'")
            line => {
              val split = line.split("\t")
              val Array(chr, pos, ref, alt) = split(variantIndex).split(":")
              split.iterator.zipWithIndex.foreach {
                case (field, index) =>
                  if (index != variantIndex)
                    ab += functions(index)(field)
              }
              val result = ab.result()
              ab.clear()
              (Variant(chr, pos.toInt, ref, alt), result)
            }
          }
          // format CHR  POS  REF  ALT
          else {
            val (chrIndex, posIndex, refIndex, altIndex) =
              (header.indexOf(vColumns(0)),
                header.indexOf(vColumns(1)),
                header.indexOf(vColumns(2)),
                header.indexOf(vColumns(3)))
            if (chrIndex < 0 || posIndex < 0 || refIndex < 0 || altIndex < 0) {
              val notFound = vColumns.flatMap(i => if (header.indexOf(i) < 0) Some(i) else None)
              fatal(s"Could not find designated identifier column(s): [${notFound.mkString(", ")}]")
            }
            line => {
              val split = line.split("\t")
              split.iterator.zipWithIndex.foreach {
                case (field, index) =>
                  if (index != chrIndex && index != posIndex && index != refIndex && index != altIndex)
                    ab += functions(index)(field)
              }
              val result = ab.result()
              ab.clear()
              (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), result)
            }
          }
        }

        val signatures = Annotations(header.flatMap(s =>
          if (!vColumns.toSet(s))
            Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
          else
            None)
          .toMap)

        val variantMap = lines.map {
          line =>
            val (variant, fields) = parseLine(line)
            (variant, bbb.add(fields))
        }
          .toMap

        internalMap = variantMap
        compressedBlocks = bbb.result()
    }
  }

  def serialize(path: String, sz: SerializerInstance) {
    val conf = new hadoop.conf.Configuration()
    val signatures = metadata(conf)
    check(sz)

    val stream = sz.serializeStream(hadoopCreate(path, conf))
      .writeObject[String]("tsv")
      .writeObject[Array[String]](headerIndex)
      .writeObject[Annotations](signatures)
      .writeObject(internalMap.size)
      .writeAll(internalMap.iterator)
      .writeObject(compressedBlocks.length)
      .writeAll(compressedBlocks.iterator)
      .close()
  }
}
