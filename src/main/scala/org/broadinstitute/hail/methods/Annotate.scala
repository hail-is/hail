package org.broadinstitute.hail.methods

import java.nio.ByteBuffer

import org.apache.spark.SparkEnv
import org.apache.spark.serializer.{Serializer, SerializerInstance}
import org.broadinstitute.hail.annotations.{Annotations, VCFSignature, SimpleSignature}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._

import scala.io.Source

object Annotate {
  def annotateSamplesFromTSV(vds: VariantDataset, path: String, root: String, sampleCol: String,
    typeMap: Map[String, String], missing: Set[String]): VariantDataset = {
    val lines = Source.fromInputStream(hadoopOpen(path, vds.sparkContext.hadoopConfiguration))
      .getLines()

    if (lines.isEmpty)
      fatal("empty annotations file")

    val header = lines
      .next()
      .split("\t")

    val cleanHeader = header.flatMap { line =>
      if (line == sampleCol)
        None
      else
        Some(line)
    }

    val functions = header.map(col => AnnotationImporters.addToMap(
      typeMap.getOrElse(col, "String"), col, missing, Set(sampleCol)))

    val sampleColIndex = header.indexOf(sampleCol)

    val sampleMap: Map[String, IndexedSeq[Option[Any]]] = {
      lines.map(line => {
        val split = line.split("\t")
        val sample = split(sampleColIndex)
        val indexedValues = split.iterator.zipWithIndex.map {
          case (field, index) =>
            functions(index)(field)
        }
          .toIndexedSeq
        (sample, indexedValues)
      })
        .toMap
    }

    println(sampleMap)

    val newSampleAnnotations = vds.sampleIds
      .map(id => sampleMap.get(id))
      .map(o => o match {
        case Some(values) =>
          Annotations(cleanHeader.zip(values).toMap)
        case None => Annotations.empty()
      })

    val signatures = header.flatMap { col =>
      if (col == sampleCol)
        None
      else
        Some(col, SimpleSignature(typeMap.getOrElse(col, "String")))
    }
      .toMap

    val sigsToAdd = {
      if (root == null)
        Annotations(signatures)
      else
        Annotations(Map(root -> Annotations(signatures)))
    }
    println(sigsToAdd)

    val localIds = vds.localSamples.map(vds.sampleIds)
    val overlap = localIds.map(sampleMap.contains)
    val missingSamples = overlap.count(_ == false)
    if (missingSamples > 0)
      println(s"WARNING: $missingSamples local samples not found in annotations file")

    vds.copy(metadata = vds.metadata.addSampleAnnotations(
      sigsToAdd, newSampleAnnotations))
  }

  def annotateVariantsFromTSV(vds: VariantDataset, path: String, root: String, typeMap: Map[String, String],
    missing: Set[String], vCols: IndexedSeq[String]): VariantDataset = {

    val reader = new TSVReader(path, vCols, typeMap, missing)

    val (header, sigs) = reader.metadata(vds.sparkContext.hadoopConfiguration, vCols)

    println(root, sigs)

    val headerBc = vds.sparkContext.broadcast(header)

    val iters = reader.getIters
//    val iteratorsBc = vds.sparkContext.broadcast(reader.getIters)
    var newvds = vds

    iters.foreach {
      variantMap =>
        println(variantMap.take(5))
        val variantMapBc = vds.sparkContext.broadcast(variantMap)
        newvds = newvds.mapAnnotations(
          (v, va) => variantMapBc.value.get(v) match {
            case (Some(values)) =>
              val va2 = Annotations(headerBc.value.zip(values).toMap)
              if (root == null)
                va ++ va2
              else
                va +(root, va2)
            case None => va
          })
    }

//    val newVds = vds.mapAnnotations(
//      (v, va) =>
//        annotationMapBc.value.get(v) match {
//          case Some(values) =>
//            val va2 = Annotations(headerBc.value.zip(values).toMap)
//            if (root == null)
//              va ++ va2
//            else
//              va +(root, va2)
//          case None =>
//            if (root == null)
//              va
//            else
//              va +(root, Annotations.empty())
//        })
//      .addVariantAnnotationSignatures({
//        if (root == null) sigs else Annotations(Map(root -> sigs))
//      })
    newvds.addVariantAnnotationSignatures(if (root == null) sigs else Annotations(Map(root -> sigs)))
  }

  def annotateVariantsFromIntervalList(vds: VariantDataset, path: String, root: String, typeMap: Map[String, String],
    iCols: IndexedSeq[String], identifier: String): VariantDataset = {


    throw new UnsupportedOperationException
  }

  def annotateVariantsFromBed(vds: VariantDataset, path: String, root: String): VariantDataset = {

    throw new UnsupportedOperationException
  }


  def annotateVariantsFromVCF(vds: VariantDataset, path: String, root: String): VariantDataset = {
    val reader = new VCFReader(path)
    val signatures = reader.signatures(vds.sparkContext.hadoopConfiguration)
    println(signatures)
    val annotationMapBc = vds.sparkContext.broadcast(reader.getInfo)
    vds.mapAnnotations {
      case (v, va) =>
        va + (root, annotationMapBc.value.getOrElse(v, Annotations(Map("info" -> Annotations.empty()))))
    }
      .addVariantAnnotationSignatures(root, signatures)
  }

  def annotateVariantsFromKryo(vds: VariantDataset, path: String, root: String): VariantDataset = {
    println("starting read")

    val reader = new KryoReader(path)

    val (source, header, signatures) = reader.metadata(vds.sparkContext.hadoopConfiguration)

    val valsBc = vds.sparkContext.broadcast(reader.getValues)

    val headerBc = vds.sparkContext.broadcast(header)

    val f: (Int, Int, SerializerInstance) => Annotations = {
      source match {
        case "vcf" =>
          (index, innerIndex, ser) =>
            val (length, bytes) = valsBc.value._2(index)
            ser.deserialize[Array[Annotations]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
              .apply(innerIndex)
        case "tsv" =>
          (index, innerIndex, ser) =>
            val (length, bytes) = valsBc.value._2(index)
            Annotations(header.zip(
              ser.deserialize[Array[IndexedSeq[Option[Any]]]](ByteBuffer.wrap(LZ4Utils.decompress(length, bytes)))
                .apply(innerIndex))
                .flatMap {
                  case (k, v) => v match {
                    case Some(value) => Some((k, value))
                    case None => None
                  }}
                  .toMap)
        case _ => throw new UnsupportedOperationException
      }
    }

    val newVds = vds.copy(rdd =
      vds.rdd.mapPartitions( iter => {
        val serializer = SparkEnv.get.serializer.newInstance()
        iter.map {
          case (v, va, gs) =>
            valsBc.value._1.get(v) match {
              case Some((index, innerIndex)) =>
                val va2 = f(index, innerIndex, serializer)
                if (root == null)
                  (v, va ++ va2, gs)
                else
                  (v, va +(root, va2), gs)
              case _ =>
                if (root == null)
                  (v, va, gs)
                else
                  (v, va +(root, Annotations.empty()), gs)
            }
        }
      })
    )
      .addVariantAnnotationSignatures(
        if (root == null)
          signatures
        else
          Annotations(Map(root -> signatures))
      )

    newVds
  }
}
