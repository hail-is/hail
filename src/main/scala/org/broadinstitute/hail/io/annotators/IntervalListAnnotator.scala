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
  @transient var signatures: Annotations = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance = null): Annotations = {
    check()
    f(v, va)
  }

  def check() {
    if (intervalList == null)
      read()
    if (signatures == null)
      signatures = metadata()
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
            case Some(result) => va.update(rooted(Annotations(Map(identifier -> result))), signatures)
            case None => va.update(Annotations.empty(), signatures)
          }
      case "Boolean" =>
        (v, va) =>
          if (intervalList.contains(v.contig, v.start))
            va.update(rooted(Annotations(Map(identifier -> true))), signatures)
          else
            va.update(rooted(Annotations(Map(identifier -> false))), signatures)
      case _ => throw new UnsupportedOperationException
    }

    rooted(Annotations(Map(identifier -> SimpleSignature(extractType))))
  }

  def read() {
    if (extractType == null)
      metadata()

    val f: Line => Interval = extractType match {
      case "String" => _.transform(line => {
        line.value.split("\t") match {
          case Array(contig, start, end, direction, target) =>
            Interval(contig, start.toInt, end.toInt, Some(target))
          case arr => fatal("Inconsistent interval file")
        }
      })
      case "Boolean" =>
        _.transform(line => {
          line.value match {
            case IntervalList.intervalRegex(contig, start_str, end_str) =>
              Interval(contig, start_str.toInt, end_str.toInt)
            case _ => fatal("Inconsistent interval file")
          }
        })
      case _ => throw new UnsupportedOperationException
    }

    intervalList = readLines(path, conf.value) { lines =>
      IntervalList(
        lines
          .filter(line => !line.value.isEmpty)
          .map(f)
          .toTraversable)
    }
  }
}
