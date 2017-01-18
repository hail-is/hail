package is.hail.expr

import java.io._

object FunctionDocumentation {

  def apply(filename: String) = {
    val sb = new StringBuilder
    sb.append(header1Rst("Functions"))

    sb.append(FunctionRegistry.generateDocumentation())

    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(sb.result())
    bw.close()
  }

  def headerRst(title: String, topChar: Option[String], botChar: Option[String]) = {
    val sb = new StringBuilder()

    topChar.foreach { ch =>
      require(ch.length == 1)
      sb.append(ch * title.length + "\n")
    }

    sb.append(title + "\n")

    botChar.foreach { ch =>
      require(ch.length == 1)
      sb.append(ch * title.length + "\n")
    }

    sb.append("\n")
    sb.result()
  }

  def header1Rst(title: String) = headerRst(title, Some("="), Some("="))

  def header2Rst(title: String) = headerRst(title, Some("-"), Some("-"))

  def header3Rst(title: String) = headerRst(title, None, Some("="))

  def header1Md(title: String) = "# " + title

  def header2Md(title: String) = "## " + title

  def header3Md(title: String) = "### " + title

  def header4Md(title: String) = "#### " + title

  def methodToRst(name: String, tt: TypeTag, fun: Fun, md: Metadata): String = { // Fixme: Should be MethodMetadata
    val sb = new StringBuilder()
    sb.append(s" - **$name")

    val argsNames = md.argNames
    val argsType = tt.xs.drop(1).map(_.toString.replaceAll("\\?", ""))
    //    require(argsNames.length == argsType.length)

    val args = argsNames.zip(argsType)
    val retType = fun.retType.toString.replaceAll("\\?", "")

    if (args.nonEmpty) {
      sb.append("(")
      sb.append(args.map { case (n, t) => s"$n: $t" }.mkString(", "))
      sb.append(")")
    }

    sb.append(s"**: *$retType*\n")

    md.docstring match {
      case Some(s) =>
        sb.append(s)
      case None =>
    }

    sb.append("\n")
    sb.result()
  }

  def functionToRst(name: String, tt: TypeTag, md: Metadata): String = ??? // Fixme: Should be FunctionMetadata

  def annotationToRst(fun: Fun): String = {
    val sb = new StringBuilder()
    fun.retType match {
      case rt: TStruct =>
        sb.append("**Annotations**\n\n")
        sb.append(rt.pretty(sb, indent = 4, printAttrs = true))
      case _ =>
    }
    sb.result()
  }
}

