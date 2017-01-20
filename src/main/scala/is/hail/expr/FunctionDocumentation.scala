package is.hail.expr

import java.io._

import is.hail.annotations.Annotation

object FunctionDocumentation {

  def makeMethodsDocs(filename: String) = {
    val methods = FunctionRegistry.methods.map { case (obj, mtds) => (obj.replaceAll("\\?", ""), mtds) }.sortBy(_._1)
    val sb = new StringBuilder

    sb.append(".. sec-methods:\n\n")
    sb.append(sectionReferenceRst("methods"))
    sb.append("\n\n")
    sb.append(header1Rst("Object Methods"))
    sb.append("\n\n")

    methods.foreach { case (obj, mtds) =>
      sb.append(sectionReferenceRst(obj.toLowerCase))
      sb.append("\n\n")
      sb.append(header2Rst(obj))
      sb.append("\n\n")
      mtds.sortBy(_._2).foreach { case (_, name, tt, f, md) =>
        val argTypes = tt.xs.drop(1)
        sb.append(methodToRst(name, argTypes, f, md))
        sb.append("\n")
        sb.append(annotationToRst(f))
        sb.append("\n")
        sb.append(argsToRst(argTypes, md))
      }
    }

    writeFile(filename, sb.result())
  }

  def makeFunctionsDocs(filename: String) = {
    val functions = FunctionRegistry.functions.sortBy(_._1)
    val sb = new StringBuilder

    sb.append(".. sec-functions:\n\n")
    sb.append(sectionReferenceRst("functions"))
    sb.append("\n\n")
    sb.append(header1Rst("Functions"))
    sb.append("\n\n")

    functions.foreach { case (name, tt, f, md) =>
      sb.append(sectionReferenceRst(name.toLowerCase))
      sb.append("\n\n")
      sb.append(functionToRst(name, tt.xs, f, md))
      sb.append("\n")
      sb.append(annotationToRst(f))
      sb.append("\n")
      sb.append(argsToRst(tt.xs, md))
    }

    writeFile(filename, sb.result())
  }

  def writeFile(filename: String, text: String) = {
    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(text)
    bw.close()
  }

  def sectionReferenceRst(title: String) = s".. _$title:"

  def headerRst(title: String, topChar: Option[String], botChar: Option[String]) = {
    val sb = new StringBuilder()

    topChar.foreach { ch =>
      require(ch.length == 1)
      sb.append(ch * math.max(title.length, 3) + "\n")
    }

    sb.append(title + "\n")

    botChar.foreach { ch =>
      require(ch.length == 1)
      sb.append(ch * math.max(title.length, 3) + "\n")
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

  def methodToRst(name: String, argTypes: Seq[Type], fun: Fun, md: Metadata): String = {
    val sb = new StringBuilder()
    sb.append(s" - **$name")

    //    require(argsTypes.length == md.args.length)
    val args = argTypes.map(_.toString.replaceAll("\\?", "")).zip(md.args)

    val retType = fun.retType.toString.replaceAll("\\?", "")

    if (args.nonEmpty) {
      sb.append("(")
      sb.append(args.map { case (t, (n, d)) => s"$n: $t" }.mkString(", "))
      sb.append(")")
    }

    sb.append(s"**: *$retType*\n")

    md.docstring match {
      case Some(s) =>
        sb.append(s"${s.split("\n").map(s => "\t" + s).mkString("\n")}")
      case None =>
    }

    sb.append("\n")
    sb.result()
  }

  def functionToRst(name: String, argTypes: Seq[Type], fun: Fun, md: Metadata) = methodToRst(name, argTypes, fun, md)

  def annotationToRst(fun: Fun): String = {
    val sb = new StringBuilder()
    fun.retType match {
      case rt: TStruct =>
        val schema = rt.toPrettyString(compact = false, printAttrs = true).split("\n").map(s => "\t" + s).mkString("\n")

        sb.append(
          s"""**Annotations**
            |
            |.. code-block:: text
            |
            |$schema
          """.stripMargin.split("\n").map(s => "\t" + s).mkString("\n")
        )
      case _ =>
    }
    sb.result()
  }

  def argsToRst(argTypes: Seq[Type], md: Metadata) = {
    val sb = new StringBuilder()
    val args = argTypes.map(_.toString.replaceAll("\\?", "")).zip(md.args)
    val argsString = args.map { case (t, (name, desc)) => s"\t - **$name** (*$t*) -- $desc" }.mkString("\n")

    if (args.nonEmpty) {
      sb.append(
        s"""**Arguments**
           |
           |$argsString
         """.stripMargin.split("\n").map(s => "\t" + s).mkString("\n")
      )
    }
    sb.append("\n\n")
    sb.result()
  }
}

