package org.broadinstitute.hail.io.annotators

import org.apache.hadoop.conf.Configuration
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotations, SimpleSignature}
import org.broadinstitute.hail.variant.{Interval, IntervalList, Variant}

import scala.io.Source

class BedAnnotator(path: String, root: String) extends VariantAnnotator {
  @transient var intervalList: IntervalList = null
  var extractType: String = null
  var name: String = null
  var f: (Variant, Annotations) => Annotations = null

  val rooted = Annotator.rootFunction(root)

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

    println(headerLines)

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
        split.length < 4
      }
      catch {
        case e: java.io.EOFException => fatal("empty bed file")
      }
    }

    extractType = if (linesBoolean) "Boolean" else "String"

    f = extractType match {
      case "String" =>
        (v, va) =>
          intervalList.query(v.contig, v.start) match {
            case Some(result) => va ++ rooted(Annotations(Map(name -> result)))
            case None => va
          }
      case "Boolean" =>
        (v, va) =>
          if (intervalList.contains(v.contig, v.start))
            va ++ rooted(Annotations(Map(name -> true)))
          else
            va
    }

    rooted(Annotations(Map(name -> SimpleSignature(if (linesBoolean) "Boolean" else "String"))))
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
        .filter(line => !line.isEmpty)
        .map(
          line => {
            val split = line.split("""\s+""", 5)
            f(split)
          })
        .toTraversable)
  }
}
