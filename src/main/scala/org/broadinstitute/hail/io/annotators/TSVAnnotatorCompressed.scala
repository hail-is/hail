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
  typeMap: Map[String, String], missing: Set[String], root: String, hConf: hadoop.conf.Configuration)
  extends VariantAnnotator {

  val conf = new SerializableHadoopConfiguration(hConf)
  @transient var internalMap: Map[Variant, (Int, Int)] = null
  @transient var compressedBlocks: Array[(Int, Array[Byte])] = null
  @transient var headerIndex: Array[String] = null
  @transient var signatures: Annotations = null

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
        va.update(rooted(Annotations(map)), signatures)
      case None => va.update(Annotations.empty(), signatures)
    }
  }

  def check(sz: SerializerInstance) {
    if (internalMap == null)
      read(sz)
    if (signatures == null)
      signatures = metadata()
  }

  def metadata(): Annotations = {
    readFile(path, conf.value)(is => {
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

  def read(serializer: SerializerInstance) {
    readLines(path, conf.value) { lines =>
      lines
          .filter(line => !line.value.isEmpty)


        val header = lines.next()
            .value
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

        val parseLine: Line => (Variant, Array[Option[Any]]) = {
          // format CHR:POS:REF:ALT
          if (vColumns.length == 1) {
            val variantIndex = header.indexOf(vColumns.head)
            fatalIf(variantIndex < 0, s"Could not find designated CHR:POS:REF:ALT column identifier '${vColumns.head}'")
            line => {
              val split = line.value.split("\t")
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
              val split = line.value.split("\t")
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
    val signatures = metadata()
    check(sz)

    val stream = writeDataFile(path, conf.value) { writer =>
      sz.serializeStream(writer)
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
}
