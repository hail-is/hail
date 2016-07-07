package org.broadinstitute.hail.methods

import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._

import scala.io.Source
import scala.language.implicitConversions

object ExportTSV {

  def parseColumnsFile(
    ec: EvalContext,
    path: String,
    hConf: hadoop.conf.Configuration): (Array[String], Array[() => Option[Any]]) = {
    val pairs = readFile(path, hConf) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(!_.isEmpty)
        .map { line =>
          val cols = line.split("\t")
          if (cols.length != 2)
            fatal("invalid .columns file.  Include 2 columns, separated by a tab")
          (cols(0), cols(1))
        }.toArray
    }

    val header = pairs.map(_._1)
    val fs = pairs.map { case (_, e) =>
      Parser.parse(e, ec)._2
    }

    (header, fs)
  }
}
