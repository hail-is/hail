package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.serializer.SerializerInstance
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotations, SimpleSignature}
import org.broadinstitute.hail.variant.{Interval, IntervalList, Variant}

import scala.io.Source


class BedAnnotator(path: String, root: String, hConf: hadoop.conf.Configuration) extends VariantAnnotator {
  // this annotator reads files in the UCSC BED spec defined here: https://genome.ucsc.edu/FAQ/FAQformat.html#format1

  val conf = new SerializableHadoopConfiguration(hConf)
  @transient var intervalList: IntervalList = null
  @transient var extractType: String = null
  @transient var name: String = null
  @transient var f: (Variant, Annotations) => Annotations = null
  @transient var signatures: Annotations = null

  val rooted = Annotator.rootFunction(root)

  def annotate(v: Variant, va: Annotations, sz: SerializerInstance): Annotations = {
    check()
    f(v, va)
  }

  def check() {
    if (intervalList == null)
      read()
    if (signatures == null)
      signatures = metadata()
  }

  def metadata(): (Annotations) = {
    readFile(path, conf.value) { reader =>
      val lines = Source.fromInputStream(reader)
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
          val split = lines.next()
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
              case Some(result) => va.update(rooted(Annotations(Map(name -> result))), signatures)
              case None => va.update(Annotations.empty(), signatures)
            }
        case "Boolean" =>
          (v, va) =>
            if (intervalList.contains(v.contig, v.start))
              va.update(rooted(Annotations(Map(name -> true))), signatures)
            else
              va.update(rooted(Annotations(Map(name -> false))), signatures)
      }

      rooted(Annotations(Map(name -> SimpleSignature(if (linesBoolean) "Boolean" else "String"))))
    }
  }

  def read() {
    readLines(path, conf.value) { lines =>
      if (extractType == null)
        metadata()

      val intervalListBuilder = IntervalList

      val f: Line => Interval = extractType match {
        case "String" => l => l.transform(line => {
          val arr = line.value.split("""\s+""")
          Interval(arr(0), arr(1).toInt, arr(2).toInt, Some(arr(3)))
        })
        case _ => l => l.transform(line => {
          val arr = line.value.split("""\s+""")
          Interval(arr(0), arr(1).toInt, arr(2).toInt)
        })
      }

      intervalList = IntervalList(
        lines.dropWhile(line =>
          line.value.startsWith("browser") ||
            line.value.startsWith("track") ||
            line.value.matches("""^\w+=("[\w\d ]+"|\d+).*"""))
          .filter(line => !line.value.isEmpty)
          .map(f)
          .toTraversable)
    }
  }
}
