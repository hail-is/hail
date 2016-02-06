package org.broadinstitute.hail.methods

import java.nio.ByteBuffer

import htsjdk.variant.variantcontext.Allele
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{VCFSignature, SimpleSignature, Annotations}
import org.broadinstitute.hail.variant.{LZ4Utils, Interval, IntervalList, Variant}
import org.broadinstitute.hail.vcf.{HtsjdkRecordReader, BufferedLineIterator}
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

import scala.io.Source


abstract class VariantAnnotator extends Serializable {

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations

  def metadata(conf: Configuration): Annotations
}

object VariantAnnotator {

  final val ITERATOR_SIZE = 2000000

  def addToMap(typeString: String, k: String, missing: Set[String], excluded: Set[String]):
  (String) => Option[Any] = {

    if (!excluded(k)) {
      typeString match {
        case "Double" =>
          (v: String) =>
            try {
              if (!missing(v))
                Some(v.toDouble)
              else
                None
            }
            catch {
              case e: java.lang.NumberFormatException =>
                fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Double in column "$k" """)
            }
        case "Int" =>
          (v: String) =>
            try {
              if (!missing(v))
                Some(v.toInt)
              else
                None
            }
            catch {
              case e: java.lang.NumberFormatException =>
                fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Int in column "$k" """)
            }
        case "Boolean" =>
          (v: String) =>
            try {
              if (!missing(v))
                Some(v.toBoolean)
              else
                None
            }
            catch {
              case e: java.lang.IllegalArgumentException =>
                fatal( s"""java.lang.IllegalArgumentException: tried to convert "$v" to Boolean in column "$k" """)
            }
        case "String" =>
          (v: String) =>
            if (!missing(v))
              Some(v)
            else
              None
        case _ =>
          fatal(
            s"""Unrecognized type "$typeString" in column "$k".  Hail supports the following types in annotations:
                |  - Double (floating point number)
                |  - Int  (integer)
                |  - Boolean
                |  - String
                |
               |  Note that the above types are case sensitive.
             """.stripMargin)
      }
    }
    else
      v => None
  }
}

class TSVAnnotatorCompressed(path: String, vColumns: IndexedSeq[String],
  typeMap: Map[String, String], missing: Set[String], root: String)
  extends VariantAnnotator {

  @transient var internalMap: Map[Variant, (Int, Int)] = null
  @transient var compressedBlocks: IndexedSeq[(Int, Array[Byte])] = null
  @transient var headerIndex: IndexedSeq[String] = null

  val rootFunction: Annotations => Annotations = {
    va =>
      if (root == null)
        va
      else
        Annotations(Map(root -> va))
  }

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
        rootFunction(Annotations(map))
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

      Annotations(header.flatMap(s =>
        if (!vColumns.toSet(s))
          Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
        else
          None)
        .toMap)
    })
  }

  def read(conf: Configuration, serializer: SerializerInstance) {
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
        if (i % 10000 == 0)
          println("line " + i + " " + indexedValues)
        val variantIndex = bbb.add(indexedValues)
        (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), variantIndex)
    }
      .toMap

    val t1 = System.nanoTime()
    println(s"read tsv took ${formatTime(t1 - t0)} seconds")

    internalMap = variantMap
    compressedBlocks = bbb.result()
  }
}

object VCFReader {

  def mapTypeMultiallelic(value: Any, sig: VCFSignature, altIndex: Int): Any = {
    value match {
      case str: String =>
        sig.typeOf match {
          case "Int" => str.toInt
          case "Double" => str.toDouble
          case _ => value
        }
      case i: IndexedSeq[_] =>
        sig.typeOf match {
          case "IndexedSeq[Int]" =>
            sig.number match {
              case "A" => IndexedSeq(i(altIndex)).map(_.asInstanceOf[String].toInt)
              case "R" =>
                val arr = IndexedSeq(i(0), i(altIndex + 1)).map(_.asInstanceOf[String].toInt)
              case "G" => throw new UnsupportedOperationException("don't currently support VCFs with 'G' in info number")
              case _ => i.map(_.asInstanceOf[String].toInt)
            }
          case "IndexedSeq[Double]" =>
            sig.number match {
              case "A" => IndexedSeq(i(altIndex)).map(_.asInstanceOf[String].toDouble)
              case "R" =>
                val arr = IndexedSeq(i(0), i(altIndex + 1)).map(_.asInstanceOf[String].toDouble)
              case "G" => throw new UnsupportedOperationException("don't currently support VCFs with 'G' in info number")
              case _ => i.map(_.asInstanceOf[String].toDouble)
            }
        }
      case _ => value
    }
  }
}

class VCFAnnotatorCompressed(path: String, root: String) extends VariantAnnotator {
  @transient var indexMap: Map[Variant, (Int, Int)] = null
  @transient var compressedBlocks: IndexedSeq[(Int, Array[Byte])] = null

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check(sz)
    indexMap.get(v) match {
      case Some((i, j)) =>
        val (length, bytes) = compressedBlocks(i)
        Annotations(Map(root -> sz.deserialize[Array[Annotations]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
          .apply(j)))
      case None => va
    }
  }

  def check(sz: SerializerInstance) {
    if (indexMap == null || compressedBlocks == null)
      read(new Configuration(), sz)
  }

  def metadata(conf: Configuration): Annotations = {
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

    Annotations(Map("info" -> infoSignatures,
      "filters" -> new SimpleSignature("Set[String]"),
      "pass" -> new SimpleSignature("Boolean"),
      "qual" -> new SimpleSignature("Double"),
      "multiallelic" -> new SimpleSignature("Boolean"),
      "rsid" -> new SimpleSignature("String")))
  }


  def read(conf: Configuration, sz: SerializerInstance) {
    val t0 = System.nanoTime()
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

    val bbb = new ByteBlockBuilder[Annotations](sz)

    indexMap = lineIterator.flatMap { line =>
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
        Iterator.single((variant, Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
          .asScala
          .mapValues(HtsjdkRecordReader.purgeJavaArrayLists)
          .toMap
          .map {
            case (k, v) => (k, HtsjdkRecordReader.mapType(v, infoSignatures.get[VCFSignature](k)))
          }),
          "qual" -> vc.getPhredScaledQual,
          "filters" -> filts,
          "pass" -> pass,
          "rsid" -> rsid))))

      }
      else {
        val ref = vc.getReference
        val alts = vc.getAlternateAlleles.asScala.filter(_ != Allele.SPAN_DEL)
        val altIndices = alts.map(vc.getAlleleIndex) // index in the VCF, used to access AD and PL fields
        alts.zipWithIndex.map { case (alt, index) =>
          (Variant(vc.getContig, vc.getStart, ref.getBaseString, alt.getBaseString),
            Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
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
              "multiallelic" -> true)))
        }
          .iterator
      }
    }
      .map { case (v, va) => (v, bbb.add(va)) }
      .toMap

    compressedBlocks = bbb.result()

    val t1 = System.nanoTime()
    println(s"read vcf took ${formatTime(t1 - t0)} seconds")
  }
}

//
//  def read(conf: Configuration) {
//    val t0 = System.nanoTime()
//    println("starting parse")
//    val lines = Source.fromInputStream(hadoopOpen(path, conf))
//      .getLines()
//    val header = lines.next()
//      .split("\t")
//    val (chrIndex, posIndex, refIndex, altIndex) =
//      (header.indexOf(vColumns(0)),
//        header.indexOf(vColumns(1)),
//        header.indexOf(vColumns(2)),
//        header.indexOf(vColumns(3)))
//
//    if (chrIndex < 0 || posIndex < 0 || refIndex < 0 || altIndex < 0) {
//      val notFound = vColumns.flatMap(i => if (header.indexOf(i) < 0) Some(i) else None)
//      fatal(s"Could not find designated identifier column(s): [${notFound.mkString(", ")}]")
//    }
//
//    val excluded = vColumns.toSet
//    val functions = header.map(col => VariantAnnotator.addToMap(typeMap.getOrElse(col, "String"),
//      col, missing, excluded)).toIndexedSeq
//
//    var i = 1
//    val variantMap = lines.map {
//      line =>
//        i += 1
//        if (i % 10000 == 0)
//          println("line " + i)
//        val split = line.split("\t")
//        val indexedValues = split.iterator.zipWithIndex.map {
//          case (field, index) =>
//            functions(index)(field)
//        }
//          .toIndexedSeq
//        (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), indexedValues)
//    }
//      .toMap
//    this.internalMap = variantMap
//
//    val t1 = System.nanoTime()
//    println(s"read tsv took ${formatTime(t1 - t0)} seconds")
//  }

/*
class TSVReaderCompressed(path: String, vColumns: IndexedSeq[String],
  typeMap: Map[String, String], missing: Set[String])
  extends AnnotationImporter[(Map[Variant, (Int, Int)], IndexedSeq[(Int, Array[Byte])])] {
  @transient var internalMap: Map[Variant, (Int, Int)] = null
  @transient var compressedBlocks: IndexedSeq[(Int, Array[Byte])] = null

  def getValues: (Map[Variant, (Int, Int)], IndexedSeq[(Int, Array[Byte])]) = {
    if (internalMap != null && compressedBlocks != null)
      (internalMap, compressedBlocks)
    else {
      read(new Configuration, SparkEnv.get.serializer.newInstance())
      (internalMap, compressedBlocks)
    }
  }

  def metadata(conf: Configuration): (IndexedSeq[String], Annotations) = {
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

      (header.flatMap(line => if (!vColumns.contains(line)) Some(line) else None).toIndexedSeq,
        Annotations(header.flatMap(s =>
          if (!vColumns.toSet(s))
            Some(s, SimpleSignature(typeMap.getOrElse(s, "String")))
          else
            None)
          .toMap))
    })
  }

  def read(conf: Configuration, serializer: SerializerInstance) {
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
        if (i % 10000 == 0)
          println("line " + i + " " + indexedValues)
        val variantIndex = bbb.add(indexedValues)
        (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), variantIndex)
    }
      .toMap

    val t1 = System.nanoTime()
    println(s"read tsv took ${formatTime(t1 - t0)} seconds")

    internalMap = variantMap
    compressedBlocks = bbb.result()
  }
}


object VCFReader {

  def mapTypeMultiallelic(value: Any, sig: VCFSignature, altIndex: Int): Any = {
    value match {
      case str: String =>
        sig.typeOf match {
          case "Int" => str.toInt
          case "Double" => str.toDouble
          case _ => value
        }
      case i: IndexedSeq[_] =>
        sig.typeOf match {
          case "IndexedSeq[Int]" =>
            sig.number match {
              case "A" => IndexedSeq(i(altIndex)).map(_.asInstanceOf[String].toInt)
              case "R" =>
                val arr = IndexedSeq(i(0), i(altIndex + 1)).map(_.asInstanceOf[String].toInt)
              case "G" => throw new UnsupportedOperationException("don't currently support VCFs with 'G' in info number")
              case _ => i.map(_.asInstanceOf[String].toInt)
            }
          case "IndexedSeq[Double]" =>
            sig.number match {
              case "A" => IndexedSeq(i(altIndex)).map(_.asInstanceOf[String].toDouble)
              case "R" =>
                val arr = IndexedSeq(i(0), i(altIndex + 1)).map(_.asInstanceOf[String].toDouble)
              case "G" => throw new UnsupportedOperationException("don't currently support VCFs with 'G' in info number")
              case _ => i.map(_.asInstanceOf[String].toDouble)
            }
        }
      case _ => value
    }
  }
}

class VCFReader(path: String) extends AnnotationImporter[Map[Variant, Annotations]] {
  @transient var internalMap: Map[Variant, Annotations] = null

  def getValues: Map[Variant, Annotations] = {
    if (internalMap != null)
      internalMap
    else {
      read(new Configuration())
      internalMap
    }
  }

  def metadata(conf: Configuration): Annotations = {
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

    Annotations(Map("info" -> infoSignatures,
      "filters" -> new SimpleSignature("Set[String]"),
      "pass" -> new SimpleSignature("Boolean"),
      "qual" -> new SimpleSignature("Double"),
      "multiallelic" -> new SimpleSignature("Boolean"),
      "rsid" -> new SimpleSignature("String")))
  }


  def read(conf: Configuration) {
    val t0 = System.nanoTime()
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

    internalMap = lineIterator.flatMap { line =>
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
        Iterator.single((variant, Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
          .asScala
          .mapValues(HtsjdkRecordReader.purgeJavaArrayLists)
          .toMap
          .map {
            case (k, v) => (k, HtsjdkRecordReader.mapType(v, infoSignatures.get[VCFSignature](k)))
          }),
          "qual" -> vc.getPhredScaledQual,
          "filters" -> filts,
          "pass" -> pass,
          "rsid" -> rsid))))

      }
      else {
        val ref = vc.getReference
        val alts = vc.getAlternateAlleles.asScala.filter(_ != Allele.SPAN_DEL)
        val altIndices = alts.map(vc.getAlleleIndex) // index in the VCF, used to access AD and PL fields
        alts.zipWithIndex.map { case (alt, index) =>
          (Variant(vc.getContig, vc.getStart, ref.getBaseString, alt.getBaseString),
            Annotations(Map[String, Any]("info" -> Annotations(vc.getAttributes
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
              "multiallelic" -> true)))
        }
          .iterator
      }
    }
      .toMap

    val t1 = System.nanoTime()
    println(s"read vcf took ${formatTime(t1 - t0)} seconds")
  }
}

class KryoReader(path: String) extends AnnotationImporter[(Map[Variant, (Int, Int)], IndexedSeq[(Int, Array[Byte])])] {
  @transient var variantIndexes: Map[Variant, (Int, Int)] = null
  @transient var compressedBytes: IndexedSeq[(Int, Array[Byte])] = null

  val inputType: String = {
    SparkEnv.get
      .serializer
      .newInstance()
      .deserializeStream(hadoopOpen(path, new Configuration()))
      .readObject[String]
  }

  def getValues: (Map[Variant, (Int, Int)], IndexedSeq[(Int, Array[Byte])]) = {
    if (variantIndexes != null)
      (variantIndexes, compressedBytes)
    else {
      read(new Configuration())
      (variantIndexes, compressedBytes)
    }
  }

  def read(conf: Configuration) {
    println("starting kryo read")
    val t0 = System.nanoTime()
    val serializerStream = SparkEnv.get
      .serializer
      .newInstance()
      .deserializeStream(hadoopOpen(path, conf))

    val inputType = serializerStream.readObject[String]
    val header = serializerStream.readObject[IndexedSeq[String]]
    val sigs = serializerStream.readObject[Annotations]
    variantIndexes = serializerStream.readObject[Map[Variant, (Int, Int)]]
    compressedBytes = serializerStream.readObject[Array[(Int, Array[Byte])]]

    val t1 = System.nanoTime()
    println(s"kryo read took ${formatTime(t1 - t0)} seconds")
  }

  def metadata(conf: Configuration): (String, IndexedSeq[String], Annotations) = {
    val serializerStream = SparkEnv.get
      .serializer
      .newInstance()
      .deserializeStream(hadoopOpen(path, conf))
    (serializerStream.readObject[String],
      serializerStream.readObject[IndexedSeq[String]],
      serializerStream.readObject[Annotations])
  }
}
*/
class BedAnnotator(path: String, root: String) extends VariantAnnotator {
  @transient var intervalList: IntervalList = null
  var extractType: String = null
  var name: String = null


  val f: (Variant, Annotations) => Annotations =
    extractType match {
      case "String" =>
        (v, va) =>
          intervalList.query(v.contig, v.start) match {
            case Some(result) => if (root == null) va +(name, result) else va +(root, Annotations(Map(name -> result)))
            case None => va
          }
      case "Boolean" =>
        (v, va) =>
          if (intervalList.contains(v.contig, v.start))
            if (root == null) va +(name, true) else va +(root, Annotations(Map(name -> true)))
          else
            va
    }

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check()
    f(v, va)
  }

  def check() {
    if (intervalList == null)
      read(new Configuration())
  }

  def metadata(conf: Configuration): (Annotations) = {
    val lines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()

    val headerLines = lines.takeWhile(line =>
      line.startsWith("browser") ||
        line.startsWith("track") ||
        line.matches("""^\w+=("[\w\d ]+"|\d+).*"""))
      .toList

    val filt = headerLines.filter(s => s.startsWith("track"))
    if (filt.length != 1)
      fatal("Invalid bed file: found 'track' in more than one header line")

    val nameR = """.*name="([\w\d\s]+)".*""".r
    name = filt.head match {
      case nameR(str) => str
      case _ => fatal("Invalid bed file: could not find identifier 'name'")
    }

    val linesBoolean = {
      try {
        val split = Source.fromInputStream(hadoopOpen(path, conf))
          .getLines()
          .take(headerLines.length + 1)
          .toList
          .last
          .split("""\s+""")
        split.length >= 4
      }
      catch {
        case e: java.io.EOFException => fatal("empty bed file")
      }
    }

    extractType = if (linesBoolean) "Boolean" else "String"

    Annotations(Map(name -> SimpleSignature(if (linesBoolean) "Boolean" else "String")))
  }

  def read(conf: Configuration) {
    println("READING THE FILE")
    val headerLength = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
      .takeWhile(line =>
        line.startsWith("browser") ||
          line.startsWith("track") ||
          line.matches("""^\w+=("[\w\d ]+"|\d+).*"""))
      .size
    val lines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()

    // skip the header
    lines.drop(headerLength)

    if (extractType == null)
      metadata(conf)

    val intervalListBuilder = IntervalList

    val f: Array[String] => Interval = extractType match {
      case "String" => arr => Interval(arr(0), arr(1).toInt, arr(2).toInt, Some(arr(3)))
      case _ => arr => Interval(arr(0), arr(1).toInt, arr(2).toInt)
    }

    intervalList = IntervalList(
      lines
        .map(
          line => {
            val split = line.split("""\s+""", 5)
            f(split)
          })
        .toTraversable)
  }
}

class IntervalListAnnotator(path: String, identifier: String, root: String)
  extends VariantAnnotator {
  @transient var intervalList: IntervalList = null
  var extractType: String = null
  var f: (Variant, Annotations) => Annotations = null


  def annotate(v: Variant, va: Annotations, sz: SerializerInstance = null): Annotations = {
    check()
    f(v, va)
  }

  def check() {
    if (intervalList == null)
      read(new Configuration())
  }

  def metadata(conf: Configuration): Annotations = {
    val firstLine = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
      .filter(line => !(line(0) == '@') && !line.isEmpty)
      .next()

    extractType = firstLine match {
      case IntervalList.intervalRegex(contig, start_str, end_str) => "Boolean"
      case line if line.split("""\s+""").length == 5 => "String"
      case _ => fatal("unsupported interval list format")
    }

    f = extractType match {
      case "String" =>
        (v, va) =>
          intervalList.query(v.contig, v.start) match {
            case Some(result) => if (root == null)
              va +(identifier, result)
            else
              va +(root, Annotations(Map(identifier -> result)))
            case None => va
          }
      case "Boolean" =>
        (v, va) =>
          if (intervalList.contains(v.contig, v.start))
            if (root == null) va +(identifier, true) else va +(root, Annotations(Map(identifier -> true)))
          else
            va
      case _ => throw new UnsupportedOperationException
    }

    if (root == null)
      Annotations(Map(identifier -> SimpleSignature(extractType)))
    else
      Annotations(Map(root -> Annotations(Map(identifier -> SimpleSignature(extractType)))))
  }

  def read(conf: Configuration) {
    if (extractType == null)
      metadata(conf)

    val f: String => Interval = extractType match {
      case "String" =>
        line =>
          val Array(contig, start, end, direction, target) = line.split("\t")
          Interval(contig, start.toInt, end.toInt, Some(target))
      case "Boolean" =>
        line =>
          line match {
            case IntervalList.intervalRegex(contig, start_str, end_str) =>
              Interval(contig, start_str.toInt, end_str.toInt)
            case _ => fatal("Inconsistent interval file")
          }
      case _ => throw new UnsupportedOperationException
    }

    intervalList = IntervalList(
      Source.fromInputStream(hadoopOpen(path, conf))
        .getLines()
        .map(f)
        .toTraversable)
  }
}
