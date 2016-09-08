package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.spi.OptionHandler
import org.kohsuke.args4j.{Option => Args4jOption, CmdLineParser}
import scala.collection.JavaConverters._
import org.json4s._
import org.json4s.jackson.JsonMethods._

object CommandMetadata extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "output file location")
    var outputFile: String = _
  }

  def description = "Parses args4j options to output json with all parameters"

  def name = "commandmeta"

  override def hidden = true

  def requiresVDS = false

  def supportsMultiallelic = true

  def newOptions = new Options

  def run(state: State, options: Options): State = {

    val orderedCommandList = ToplevelCommands.commands.values.toIndexedSeq.flatMap {
      case (y: SuperCommand) => y.subcommands.values
      case (x: Command) => IndexedSeq(x)
      case _ => None
    }.sortWith((x1, x2) => x1.name < x2.name)

    def synopsis(cmdName: String, parser: CmdLineParser): String = {
      val options = parser.getOptions.iterator().asScala
      val arguments = parser.getArguments.iterator().asScala
      val sb = new StringBuilder

      def reformatNameMeta(nm: String): String = {
        val x = nm.replace("(", "").replace(")", "").split("\\s+")
        val name = x.filter(_.startsWith("-")).reduceLeft((a, b) => if (a.length > b.length) a else b)
        val meta = x.filterNot(_.startsWith("-"))

        if (meta.isEmpty) name else name + " " + meta(0)
      }

      sb.append(cmdName)

      options.toIndexedSeq.filter(!_.option.hidden).sortBy(_.option.required).foreach { oh =>
        sb.append(' ')
        if (!oh.option.required)
          sb.append('[')
        sb.append(reformatNameMeta(oh.getNameAndMeta(null, parser.getProperties)))
        if (oh.option.isMultiValued) {
          sb.append(" ...")
        }
        if (!oh.option.required())
          sb.append(']')
      }

      arguments.toIndexedSeq.filter(!_.option.hidden).foreach { ah =>
        sb.append(' ')
        sb.append(ah.option.usage)
      }

      sb.result()
    }

    def optionToJSON(oh: OptionHandler[_]) = {
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
    }

    val optionsJSON = {
      val parserGlobal = new CmdLineParser(new Main.Options)
      val globalOptions = parserGlobal.getOptions.iterator().asScala

      pretty(JObject(List(JField("commands",JObject(orderedCommandList.map { c =>
        val parser = new CmdLineParser(c.newOptions)
        val options = parser.getOptions.iterator().asScala

        JField(c.name, JObject(List(
          JField("synopsis", JString(synopsis(c.name, parser))),
          JField("description", JString(c.description)),
          JField("hidden", JBool(c.hidden)),
          JField("requiresVDS", JBool(c.requiresVDS)),
          JField("supportsMultiallelic", JBool(c.supportsMultiallelic)),
          JField("options", JObject(options.map(optionToJSON).toList))
      )))}.toList)),

      JField("global", JObject( JField("options", JObject(globalOptions.map(optionToJSON).toList)))))))
    }

    state.hadoopConf.writeTextFile(options.outputFile) { out => out.write(optionsJSON) }

    state
  }
}
