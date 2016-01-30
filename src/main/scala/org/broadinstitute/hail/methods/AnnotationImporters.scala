package org.broadinstitute.hail.methods

import htsjdk.variant.variantcontext.Allele
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{VCFSignature, SimpleSignature, Annotations}
import org.broadinstitute.hail.variant.{IntervalList, Variant}
import org.broadinstitute.hail.vcf.{HtsjdkRecordReader, BufferedLineIterator}
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

import scala.io.Source

object AnnotationImporters {

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

class TSVReader(path: String, vColumns: IndexedSeq[String],
  typeMap: Map[String, String], missing: Set[String]) extends Serializable {
//  @transient var internalMap: Map[Variant, IndexedSeq[Option[Any]]] = null
  @transient var mapIterator: Iterator[Map[Variant, IndexedSeq[Option[Any]]]] = null

//
//  def getMap: Map[Variant, IndexedSeq[Any]] = {
//    if (internalMap != null)
//      internalMap
//    else {
//      read(new Configuration)
//      println(internalMap.size)
//      println(internalMap.take(5))
//      internalMap
//    }
//  }

  def getIters: Iterator[Map[Variant, IndexedSeq[Any]]] = {
    if (mapIterator != null)
      mapIterator
    else {
      read(new Configuration)
      mapIterator
    }
  }


  def metadata(conf: Configuration, vColumns: IndexedSeq[String]): (IndexedSeq[String], Annotations) = {
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

  def read(conf: Configuration) {
    val t0 = System.nanoTime()
    println("starting parse")
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

    val excluded = vColumns.toSet
    val functions = header.map(col => AnnotationImporters.addToMap(typeMap.getOrElse(col, "String"), col, missing, excluded)).toIndexedSeq

    var i = 1
    val mapIterator = (0 until 1000).iterator
      .flatMap { n =>
        if (!lines.hasNext)
          None
        else
        Some(lines.take(AnnotationImporters.ITERATOR_SIZE).map {
          line =>
            i += 1
            if (i % 1000 == 0)
              println("line " + i)
            val split = line.split("\t")
            val indexedValues = split.iterator.zipWithIndex.map {
              case (field, index) =>
                functions(index)(field)
            }
              .toIndexedSeq
            (Variant(split(chrIndex), split(posIndex).toInt, split(refIndex), split(altIndex)), indexedValues)
        }
          .toMap)
      }
    this.mapIterator = mapIterator
//    this.internalMap = lines.map {
//      line =>
//        i += 1
//        if (i % 1000 == 0)
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

    val t1 = System.nanoTime()
    println(s"read tsv took ${formatTime(t1 - t0)} seconds")
  }
}


class VCFReader(path: String) {
  @transient var vMap: Map[Variant, Annotations] = null

  def getInfo: Map[Variant, Annotations] = {
    if (vMap != null)
      vMap
    else {
      read(new Configuration())
      vMap
    }
  }

  def signatures(conf: Configuration): Annotations = {
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

    vMap = lineIterator.flatMap { line =>
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
                case (k, v) => (k, HtsjdkRecordReader.mapTypeMultiallelic(v, infoSignatures.get[VCFSignature](k),
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


class KryoReader(path: String) extends Serializable {
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


class IntervalAnnotationReader(path: String, iCols: IndexedSeq[String], typeMap: Map[String, String],
  identifier: String)
  extends Serializable {
  @transient var intervalList: IntervalList = null

  def getIntervalList: IntervalList = {
    if (intervalList != null)
      intervalList
    else {
      read(new Configuration)
      intervalList
    }
  }

  def read(conf: hadoop.conf.Configuration): Annotations = {
    val lines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
    val header = lines.next().split("\t")
    val (chrIndex, startIndex, endIndex) =
      (header.indexOf(iCols(0)),
        header.indexOf(iCols(1)),
        header.indexOf(iCols(2)))

    if (chrIndex < 0 || startIndex < 0 || endIndex < 0) {
      val notFound = iCols.flatMap(i => if (header.indexOf(i) < 0) Some(i) else None)
      fatal(s"Could not find designated identifier column(s): [${notFound.mkString(", ")}]")
    }

    val ignore = header.toSet -- iCols

    val functions = {
      if (identifier == null) {
        if (header.length != 4)
          fatal(s"Cannot parse interval TSV: if no identifier is set, hail expects a column for ")
      }
    }
    throw new UnsupportedOperationException
  }
}
