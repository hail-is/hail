package org.broadinstitute.hail.driver

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.broadinstitute.hail.Utils._
import org.apache.hadoop
import org.broadinstitute.hail.io.bgen.BgenLoader
import org.kohsuke.args4j.Argument

import scala.collection.JavaConverters._

object IndexBGEN extends Command {
  def name = "indexbgen"

  def description = "Create an index for one or more BGEN files.  `importbgen' cannot run without these indexes."

  class Options extends BaseOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false


  def run(state: State, options: Options): State = {

    val inputs = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

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
