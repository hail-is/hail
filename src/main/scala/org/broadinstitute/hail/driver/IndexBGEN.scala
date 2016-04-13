package org.broadinstitute.hail.driver

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.spark.SparkEnv
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.BgenLoader
import org.apache.hadoop
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object IndexBGEN extends Command {
  def name = "indexbgen"

  def description = "Create an index for one or more BGEN files.  `importbgen' cannot run without these indexes."

  class Options extends BaseOptions {
    @Argument(usage = "<file>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {

    val inputs = options.arguments.asScala
      .iterator
      .flatMap { arg =>
        val fss = hadoopGlobAndSort(arg, state.hadoopConf)
        val files = fss.map(_.getPath.toString)
        if (files.isEmpty)
          warn(s"`$arg' refers to no files")
        files
      }.toArray

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen")) {
        fatal(s"unknown input file: $input")
      }
    }


    class SerializableHadoopConfiguration(@transient var value: hadoop.conf.Configuration) extends Serializable {
      private def writeObject(out: ObjectOutputStream) {
        out.defaultWriteObject()
        value.write(out)
      }

      private def readObject(in: ObjectInputStream) {
        value = new hadoop.conf.Configuration(false)
        value.readFields(in)
      }
    }

    val sHC = new SerializableHadoopConfiguration(state.hadoopConf)

    state.sc.parallelize(inputs).foreach { in =>
        BgenLoader.index(sHC.value, in)
    }

    info(s"Number of BGEN files indexed: ${inputs.length}")

    state
  }
}
