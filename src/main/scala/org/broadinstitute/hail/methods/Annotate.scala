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

    val functions = header.map(col => VariantAnnotator.addToMap(
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

    val newSampleAnnotations = vds.sampleIds
      .map(id => sampleMap.get(id))
      .map(o => o match {
        case Some(values) =>
          if (root != null)
            Annotations(Map(root -> Annotations(cleanHeader.zip(values).toMap)))
          else
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

    val localIds = vds.localSamples.map(vds.sampleIds)
    val overlap = localIds.map(sampleMap.contains)
    val missingSamples = overlap.count(_ == false)
    if (missingSamples > 0)
      println(s"WARNING: $missingSamples local samples not found in annotations file")

    vds.copy(metadata = vds.metadata.addSampleAnnotations(
      sigsToAdd, newSampleAnnotations))
  }
}