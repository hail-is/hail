package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotations, SimpleSignature}
import org.broadinstitute.hail.variant.{Interval, IntervalList, Variant}

import scala.io.Source

class IntervalListAnnotator(path: String, identifier: String, root: String, hConf: hadoop.conf.Configuration)
  extends VariantAnnotator {

  val conf = new SerializableHadoopConfiguration(hConf)
  @transient var intervalList: IntervalList = null
  @transient var extractType: String = null
  @transient var f: (Variant, Annotations) => Annotations = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance = null): Annotations = {
    check()
    f(v, va)
  }

  def check() {
    if (intervalList == null)
      read()
  }

  def metadata(): Annotations = {
    val firstLine = readFile(path, conf.value) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !(line(0) == '@') && !line.isEmpty)
        .next()
    }

    extractType = firstLine match {
      case IntervalList.intervalRegex(contig, start_str, end_str) => "Boolean"
      case line if line.split("""\s+""").length == 5 => "String"
      case _ => fatal("unsupported interval list format")
    }

    f = extractType match {
      case "String" =>
        (v, va) =>
          intervalList.query(v.contig, v.start) match {
            case Some(result) => va ++ rooted(Annotations(Map(identifier -> result)))
            case None => va
          }
      case "Boolean" =>
        (v, va) =>
          if (intervalList.contains(v.contig, v.start))
            va ++ rooted(Annotations(Map(identifier -> true)))
          else
            va
      case _ => throw new UnsupportedOperationException
    }

    rooted(Annotations(Map(identifier -> SimpleSignature(extractType))))
  }

  def read() {
    if (extractType == null)
      metadata()

    val f: String => Interval = extractType match {
      case "String" =>
        line =>
          val Array(contig, start, end, direction, target) = line.split("\t")
          Interval(contig, start.toInt, end.toInt, Some(target))
      case "Boolean" =>
        line => line match {
            case IntervalList.intervalRegex(contig, start_str, end_str) =>
              Interval(contig, start_str.toInt, end_str.toInt)
            case _ => fatal("Inconsistent interval file")
          }
      case _ => throw new UnsupportedOperationException
    }

    intervalList = readFile(path, conf.value) { reader =>
      IntervalList(Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.isEmpty)
        .map(f)
        .toTraversable)
    }
  }
}
