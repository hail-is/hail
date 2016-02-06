package org.broadinstitute.hail.methods

import htsjdk.variant.variantcontext.Allele
import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{VCFSignature, SimpleSignature, Annotations}
import org.broadinstitute.hail.variant.{LZ4Utils, Variant}
import org.broadinstitute.hail.vcf.{HtsjdkRecordReader, BufferedLineIterator}
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
import scala.reflect.ClassTag


class TSVCompressor(path: String, vColumns: IndexedSeq[String],
  typeMap: Map[String, String], missing: Set[String]) extends Serializable {

  def parse(conf: Configuration, serializer: SerializerInstance):
  (IndexedSeq[String], Annotations, Array[(Variant, (Int, Int))], Array[(Int, Array[Byte])]) = {
    println("starting parse")
    val t0 = System.nanoTime()
    val lines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
    val header = lines.next()
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

    val signatures = Annotations(header.flatMap(s =>
      if (!vColumns.toSet(s))
        Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
      else
        None)
      .toMap)

    val excluded = vColumns.toSet
    val functions = header.map(col => VariantAnnotator.addToMap(typeMap.getOrElse(col, "String"), col, missing, excluded)).toIndexedSeq
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
        if (i % 100000 == 0)
          println("line " + i + " " + indexedValues)
        val variantIndex = bbb.add(indexedValues)
        (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), variantIndex)
    }
      .toArray

    val t1 = System.nanoTime()
    println(s"read tsv took ${formatTime(t1 - t0)} seconds")

    (header.flatMap(line => if (!vColumns.contains(line)) Some(line) else None).toIndexedSeq,
      signatures,
      variantMap,
      bbb.result())
  }
}


class VCFCompressor(path: String) {
  def parse(conf: Configuration, serializer: SerializerInstance):
  (Annotations, Map[Variant, (Int, Int)], Array[(Int, Array[Byte])]) = {
    val t0 = System.nanoTime()
    println("starting parse")
    val lineIterator = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()

    val headerLines = lineIterator
      .takeWhile { line => line(0) == '#' }
      .toArray

    val codec = new htsjdk.variant.vcf.VCFCodec()

    val header = codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
      .getHeaderValue
      .asInstanceOf[htsjdk.variant.vcf.VCFHeader]

    val infoSignatures = Annotations(header
      .getInfoHeaderLines
      .toList
      .map(line => (line.getID, VCFSignature.parse(line)))
      .toMap)

    val signatures = Annotations(Map("info" -> infoSignatures,
      "filters" -> new SimpleSignature("Set[String]"),
      "pass" -> new SimpleSignature("Boolean"),
      "qual" -> new SimpleSignature("Double"),
      "multiallelic" -> new SimpleSignature("Boolean"),
      "rsid" -> new SimpleSignature("String")))

    val bbb = new ByteBlockBuilder[Annotations](serializer)


    val variantMap = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
      .dropWhile { line => line(0) == '#' }
      .flatMap { line =>
        val vc = codec.decode(line)
        val pass = vc.filtersWereApplied() && vc.getFilters.size() == 0
        val filts = {
          if (vc.filtersWereApplied && vc.isNotFiltered)
            Set("PASS")
          else
            vc.getFilters.asScala.toSet
        }
        val rsid = vc.getID
        if (vc.isBiallelic) {
          val variant = Variant(vc.getContig, vc.getStart, vc.getReference.getBaseString,
            vc.getAlternateAllele(0).getBaseString)
          val anno = Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
            .asScala
            .mapValues(HtsjdkRecordReader.purgeJavaArrayLists)
            .toMap
            .map {
              case (k, v) => (k, HtsjdkRecordReader.mapType(v, infoSignatures.get[VCFSignature](k)))
            }),
            "qual" -> vc.getPhredScaledQual,
            "filters" -> filts,
            "pass" -> pass,
            "rsid" -> rsid))
          val variantIndex = bbb.add(anno)
          Iterator.single((variant, variantIndex))
        }
        else {
          val ref = vc.getReference
          val alts = vc.getAlternateAlleles.asScala.filter(_ != Allele.SPAN_DEL)
          val altIndices = alts.map(vc.getAlleleIndex) // index in the VCF, used to access AD and PL fields
          alts.zipWithIndex.map { case (alt, index) =>
            val anno = Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
              .asScala
              .mapValues(HtsjdkRecordReader.purgeJavaArrayLists)
              .toMap
              .map {
                case (k, v) => (k, VCFReader.mapTypeMultiallelic(v, infoSignatures.get[VCFSignature](k),
                  index))
              }),
              "qual" -> vc.getPhredScaledQual,
              "filters" -> filts,
              "pass" -> pass,
              "rsid" -> rsid,
              "multiallelic" -> true))
            val variantIndex = bbb.add(anno)
            (Variant(vc.getContig, vc.getStart, ref.getBaseString, alt.getBaseString), variantIndex)
          }
            .iterator
        }
      }
      .toMap

    val t1 = System.nanoTime()
    println(s"read vcf took ${formatTime(t1 - t0)} seconds")

    (signatures, variantMap, bbb.result())
  }
}

class ByteBlockBuilder[T](ser: SerializerInstance, blockSize: Int = 10)(implicit ev: ClassTag[T]) {
  val compressedArrays = mutable.ArrayBuilder.make[(Int, Array[Byte])]
  var buildingBlock = mutable.ArrayBuilder.make[T]
  var count = 0
  var index = 0


  def add(element: T): (Int, Int) = {
    buildingBlock += element
    val innerIndex = count
    count += 1

    if (count >= blockSize) {
      val res = ser.serialize(buildingBlock.result()).array()
      compressedArrays += ((res.length, LZ4Utils.compress(res)))
      buildingBlock.clear()
      count = 0
      index += 1
      (index - 1, innerIndex)
    }
    else
      (index, innerIndex)
  }

  def result(): Array[(Int, Array[Byte])] = {
    val res = ser.serialize(buildingBlock.result()).array()
    compressedArrays += ((res.length, LZ4Utils.compress(res)))
    val compressedResult = compressedArrays.result()
    compressedArrays.clear()
    compressedResult
  }
}
