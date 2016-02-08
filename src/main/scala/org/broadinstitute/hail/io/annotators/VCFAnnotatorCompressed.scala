package org.broadinstitute.hail.io.annotators

import htsjdk.variant.variantcontext.Allele
import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotations, SimpleSignature, VCFSignature}
import org.broadinstitute.hail.variant.{LZ4Utils, Variant}
import org.broadinstitute.hail.vcf.{BufferedLineIterator, HtsjdkRecordReader}
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.io.Source
import java.nio.ByteBuffer

object VCFAnnotatorCompressed {
  def mapTypeSplit(value: Any, sig: VCFSignature, altIndex: Int): Any = {
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
              case "G" =>
                throw new UnsupportedOperationException("don't currently support VCFs with 'G' in info number")
              case _ => i.map(_.asInstanceOf[String].toInt)
            }
          case "IndexedSeq[Double]" =>
            sig.number match {
              case "A" => IndexedSeq(i(altIndex)).map(_.asInstanceOf[String].toDouble)
              case "R" =>
                val arr = IndexedSeq(i(0), i(altIndex + 1)).map(_.asInstanceOf[String].toDouble)
              case "G" =>
                throw new UnsupportedOperationException("don't currently support VCFs with 'G' in info number")
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

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check(sz)
    indexMap.get(v) match {
      case Some((i, j)) =>
        val (length, bytes) = compressedBlocks(i)
        va  ++ Annotations(Map(root -> sz.deserialize[Array[Annotations]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
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

    rooted(Annotations(Map("info" -> infoSignatures,
      "filters" -> new SimpleSignature("Set[String]"),
      "pass" -> new SimpleSignature("Boolean"),
      "qual" -> new SimpleSignature("Double"),
      "wasSplit" -> new SimpleSignature("Boolean"),
      "rsid" -> new SimpleSignature("String"))))
  }


  def read(conf: Configuration, sz: SerializerInstance) {

    val headerLines = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
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

    indexMap = Source.fromInputStream(hadoopOpen(path, conf))
      .getLines()
      .drop(headerLines.length)
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
                  case (k, v) => (k, VCFAnnotatorCompressed.mapTypeSplit(v, infoSignatures.get[VCFSignature](k),
                    index))
                }),
                "qual" -> vc.getPhredScaledQual,
                "filters" -> filts,
                "pass" -> pass,
                "rsid" -> rsid,
                "wasSplit" -> true)))
          }
            .iterator
        }
      }
      .map { case (v, va) => (v, bbb.add(va)) }
      .toMap

    compressedBlocks = bbb.result()
  }

  def serialize(path: String, sz: SerializerInstance) {
    val conf = new Configuration()
    val signatures = metadata(conf)
    check(sz)

    val stream = sz.serializeStream(hadoopCreate(path, conf))
      .writeObject[String]("vcf")
      .writeObject[IndexedSeq[String]](null)
      .writeObject[Annotations](signatures)
      .writeObject(indexMap)
      .writeObject(compressedBlocks)
      .close()
  }
}
