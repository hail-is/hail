package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineParser}
import scala.collection.JavaConverters._
import org.json4s._
import org.json4s.jackson.JsonMethods._

object BuildDocs extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "output file location")
    var outputFile: String = _
  }

  def description = "Builds documentation"

  def name = "builddocs"

  override def hidden = true

  def requiresVDS = false

  def supportsMultiallelic = true

  def newOptions = new Options

  def run(state: State, options: Options): State = {

    val orderedCommandList = ToplevelCommands.commands.flatMap{
      case (s, y: SuperCommand) => y.subcommands.toIndexedSeq
      case (s, x: Command) => IndexedSeq((s, x))
      case _ => None
    }.toIndexedSeq.sortWith((x1, x2) => x1._2.name < x2._2.name)

    def synopsis(cmdName: String, parser: CmdLineParser): String = {
      val options = parser.getOptions.iterator().asScala
      val sb = new StringBuilder

      sb.append(cmdName)
      options.foreach { oh =>
        sb.append(' ')
        if (!oh.option.required)
          sb.append('[')
        sb.append(oh.getNameAndMeta(null, parser.getProperties))
        if (oh.option.isMultiValued) {
          sb.append(" ...")
        }
        if (!oh.option.required())
          sb.append(']')
      }

      sb.result()
    }

    val commandOptionsJSON = {
      pretty(JObject(orderedCommandList.map { case (s, c) =>
        val parser = new CmdLineParser(c.newOptions)
        val options = parser.getOptions.iterator().asScala

        JField(c.name, JObject(List(
          JField("synopsis", JString(synopsis(c.name, parser))),
          JField("description", JString(c.description)),
          JField("hidden", JBool(c.hidden)),
          JField("requiresVDS", JBool(c.requiresVDS)),
          JField("supportsMultiallelic", JBool(c.supportsMultiallelic)),

          JField("options", JObject(options.map { oh =>
            val optionName = oh.option.toString

            JField(optionName, JObject(List(
              JField("required", JBool(oh.option.required())),
              JField("type", JString(oh.setter.getType.getSimpleName)),
              JField("defaultValue", JString(oh.printDefaultValue())),
              JField("hidden", JBool(oh.option.hidden())),
              JField("usage", JString(oh.option.usage())),
              JField("multiArgs", JBool(oh.option.isMultiValued)),
              JField("metaVar", JString(oh.getMetaVariable(null)))
            )))
          }.toList)))))
      }.toList))
    }

    writeTextFile(options.outputFile, state.hadoopConf) { out => out.write(commandOptionsJSON) }

    state
  }
}
