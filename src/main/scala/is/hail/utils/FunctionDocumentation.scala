package is.hail.utils

import java.io.{BufferedWriter, File, FileWriter}

import is.hail.expr._
import is.hail.expr.typ._

object RstUtils {
  def sectionReference(title: String) = s".. _${ title.toLowerCase }:\n\n"

  def header(title: String, topChar: Option[String], botChar: Option[String]) = {
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

  def header1(title: String) = header(title, Some("="), Some("="))

  def header2(title: String) = header(title, Some("-"), Some("-"))

  def header3(title: String) = header(title, None, Some("="))
}

object Argument {
  def apply(name: String, typ: String, desc: String): Argument = Argument(name, typ, Option(desc))

  def apply(name: String, typ: String): Argument = Argument(name, typ, None)
}

case class Argument(name: String, typ: String, desc: Option[String])

object DocumentationEntry {
  val symbolRegex = """\p{javaJavaIdentifierStart}\p{javaJavaIdentifierPart}*""".r

  def apply(name: String, tt: TypeTag, f: Fun, md: MetaData, varArgs: Boolean = false): DocumentationEntry = {
    val namePretty = name.replaceAll("\\*", "\\\\*")

    val isMethod = tt.isInstanceOf[MethodType]
    val isField = tt.isInstanceOf[FieldType]

    val argTypes = (if (isMethod || isField) tt.xs.tail else tt.xs).map { t =>
      t match {
        case TFunction(_, _) => t.toString.replaceAll("[?!]", "").replaceFirst("\\(", "").replaceAll("\\) => ", " => ")
        case _ => t.toString.replaceAll("[?!]", "")
      }
    }.toArray

    val nArgs = argTypes.length
    val argNames = if (md.args.nonEmpty) md.args.map(_._1).toArray else ('a' to 'z').take(nArgs).map(_.toString).toArray
    val argDescs = if (md.args.nonEmpty) md.args.map(_._2).toArray else Array.fill[String](nArgs)(null)
    val args = (argNames, argTypes, argDescs).zipped.map { case (n, t, d) => Argument(n, t, d) }

    val objType = if (isMethod || isField) Some(tt.xs.head) else None

    val category = tt match {
      case x: MethodType => "method"
      case x: FieldType => "field"
      case x: FunType =>
        if (symbolRegex.findFirstIn(name).isEmpty && argTypes.length == 2)
          "symbol"
        else
          "function"
      case _ => fatal(s"Did not recognize TypeTag ${ tt.toString }")
    }

    DocumentationEntry(namePretty, category, objType, f.retType, args, md.docstring.orNull, varArgs = varArgs)
  }

  def apply(name: String, category: String, objType: Option[Type], retType: Type,
    args: Array[Argument], docstring: String): DocumentationEntry = DocumentationEntry(name, category, objType, retType, args, docstring, varArgs = false)
}

case class DocumentationEntry(name: String, category: String, objType: Option[Type], retType: Type,
  args: Array[Argument], docstring: String, varArgs: Boolean) {

  val (nLinesDocstring, docstringPretty) = formatDocstring

  val isMethod = category == "method"
  val isField = category == "field"
  val isSymbol = category == "symbol"
  val isFunction = category == "function"
  require(Array(isMethod, isField, isFunction, isSymbol).map(_.toInt).sum == 1)

  val objCategory = {
    objType match {
      case Some(ot) => ot match {
        case TAggregable(_, req) => Some("Aggregable")
        case TAggregableVariable(_, _) => Some("Aggregable")
        case TArray(_, req) => Some("Array")
        case TSet(_ , req) => Some("Set")
        case TDict(_, _, req) => Some("Dict")
        case _ => Some(ot.toString.replaceAll("[?!]", ""))
      }
      case None => None
    }
  }

  def nArgs = args.length

  def hasArgsDesc = args.exists(a => a.desc.isDefined)

  def isSlice = name.contains("[") && name.contains("]")

  def hasAnnotation = retType.isInstanceOf[TStruct]

  val retTypePretty = retType.toString.replaceAll("[?!]", "").replaceAll("Empty", "Struct")
  val objTypePretty = objType.getOrElse("").toString.replaceAll("[?!]", "")

  def formatDocstring: (Int, String) = {
    Option(docstring) match {
      case Some(ds) =>
        val strippedDocstring = FunctionDocumentation.reformatMultiLineString(ds)
        val split = strippedDocstring.split("\n")
        val nLines = split.length
        val shiftLines = (nLines > 1 || nArgs != 0).toInt

        val newDocstring = split.map { line =>
          "\t" * shiftLines + line
        }.mkString("\n")

        (nLines, newDocstring)

      case None => (0, "")
    }
  }

  def emitArgsDescription: String = {
    val sb = new StringBuilder()

    val argsString = args.zipWithIndex.map { case (a, i) =>
      if (!varArgs || i != nArgs - 1)
        s"\t - **${ a.name }** (*${ a.typ }*) -- ${ a.desc.getOrElse("") }"
      else
        s"\t - **${ a.name }** (*${ a.typ }\\**) -- ${ a.desc.getOrElse("") }"
    }.mkString("\n")

    sb.append(
      s"""**Arguments**
         |
         |$argsString
       """.stripMargin.split("\n").map(s => "\t" + s).mkString("\n")
    )

    sb.append("\n\n")
    sb.result()
  }

  def emitHeaderSymbol = {
    require(nArgs == 2)
    val arg1 = args(0)
    val arg2 = args(1)

    s" - **(${ arg1.name }: ${ arg1.typ }) $name (${ arg2.name }: ${ arg2.typ })**: *$retTypePretty*"
  }

  def emitHeaderSlice: String = {
    val nameSubstituted =
      if (name == "[]") {
        require(nArgs == 1)
        "[" + args(0).name + "]"
      }
      else {
        require(nArgs == name.split("\\*").length - 1)
        name.split("\\*").zipWithIndex.map { case (s, i) => if (i < nArgs) s + args(i).name else s }.mkString("")
      }
    s" - **$nameSubstituted**: *$retTypePretty*"
  }

  def emitHeader = {
    val sb = new StringBuilder()
    sb.append(s" - **$name")

    if (!isField) {
      sb.append("(")
      sb.append(args.zipWithIndex.map { case (a, i) =>
        if (!varArgs || i != nArgs - 1)
          s"${ a.name }: ${ a.typ }"
        else
          s"${ a.name }: ${ a.typ }*"
      }.mkString(", "))
      sb.append(")")
    }

    sb.append(s"**: *$retTypePretty*")
    sb.result()
  }

  def emit = {
    val sb = new StringBuilder()

    if (isSlice)
      sb.append(emitHeaderSlice)
    else if (!isSymbol)
      sb.append(emitHeader)
    else
      sb.append(emitHeaderSymbol)

    if (nLinesDocstring == 0)
      sb.append("\n\n")
    else if (nLinesDocstring == 1 && !hasArgsDesc && !hasAnnotation)
      sb.append(s" -- $docstringPretty\n\n")
    else
      sb.append(s"\n\n$docstringPretty\n\n")

    if (hasArgsDesc)
      sb.append(emitArgsDescription)

    sb.result()
  }
}

object FunctionDocumentation {

  import RstUtils._

  val namesToSkip = Set("fromInt", "!", "-")

  // hack for functions not in registry
  val addtlEntries = Array(
    DocumentationEntry("select", "function", None, TStruct(),
      Array(Argument("s", "Struct", "Struct to select fields from."),
        Argument("identifiers", "String", "Field names to select from ``s``. Multiple arguments allowed.")),
      """
      Return a new ``Struct`` with a subset of fields.

      .. code-block:: text
          :emphasize-lines: 2

          let s = {gene: "ACBD", function: "LOF", nHet: 12} in select(s, gene, function)
          result: {gene: "ACBD", function: "LOF"}
      """, varArgs = true),
    DocumentationEntry("drop", "function", None, TStruct(),
      Array(Argument("s", "Struct", "Struct to drop fields from."),
        Argument("identifiers", "String", "Field names to drop from ``s``. Multiple arguments allowed.")),
      """
      Return a new ``Struct`` with the subset of fields not matching ``identifiers``.

      .. code-block:: text
          :emphasize-lines: 2

          let s = {gene: "ACBD", function: "LOF", nHet: 12} in drop(s, gene, function)
          result: {nHet: 12}
      """, varArgs = true),
    DocumentationEntry("merge", "function", None, TStruct(),
      Array(Argument("s1", "Struct"), Argument("s2", "Struct")),
      """
      Create a new ``Struct`` with all fields in ``s1`` and ``s2``.

      .. code-block:: text
          :emphasize-lines: 2

          let s1 = {gene: "ACBD", function: "LOF"} and s2 = {a: 20, b: "hello"} in merge(s1, s2)
          result: {gene: "ACBD", function: "LOF", a: 20, b: "hello"}
      """),
    DocumentationEntry("ungroup", "function", None, TStruct(),
      Array(Argument("s", "Struct", "Struct to ungroup fields from."),
        Argument("identifier", "String", "Field name to ungroup from ``s``. The field type must be a Struct."),
        Argument("mangle", "Boolean", "Rename ungrouped field names as ``identifier.child``.")),
      """
      Return a new ``Struct`` where the subfields of the field given by ``identifier`` are lifted as fields in ``s``.

      .. code-block:: text
          :emphasize-lines: 2

          let s = {gene: "ACBD", info: {A: 5, B: 3}, function: "LOF"} in ungroup(s, info, true)
          result: {gene: "ACBD", function: "LOF", info.A: 5, info.B: 3}
      """, varArgs = false),
    DocumentationEntry("group", "function", None, TStruct(),
      Array(Argument("s", "Struct", "Struct to group fields from."),
        Argument("dest", "String", "Location to place new struct field in ``s``."),
        Argument("identifiers", "String", "Field names to group from ``s``. Multiple arguments allowed.")),
      """
      Return a new ``Struct`` where the fields given by ``identifiers`` are inserted into a new field in ``s`` with name ``dest`` and type ``Struct``.

      .. code-block:: text
          :emphasize-lines: 2

          let s = {gene: "ACBD", function: "LOF", nhet: 6} in group(s, grouped_field, gene, function)
          result: {nhet: 6, grouped_field: {gene: "ACBD", function: "LOF"}}
      """, varArgs = true),
    DocumentationEntry("index", "function", None, TDict(TString(), TStruct()),
      Array(Argument("structs", "Array[Struct]"),
        Argument("identifier", "String")),
      """
      Returns a Dict keyed by the string field ``identifier`` of each ``Struct`` in the ``Array`` and values that are structs with the remaining fields.

      .. code-block:: text
          :emphasize-lines: 6

          let a = [{PLI: 0.998, genename: "gene1", hits_in_exac: 1},
                   {PLI: 0.0015, genename: "gene2", hits_in_exac: 10},
                   {PLI: 0.9045, genename: "gene3", hits_in_exac: 2}] and
              d = index(a, genename) in global.gene_dict["gene1"]

          result: {PLI: 0.998, hits_in_exac: 1}
      """)
  )

  private val entries = FunctionRegistry.getRegistry()
    .flatMap { case (name, fns) => fns.map { case (tt, f, md) => DocumentationEntry(name, tt, f, md) } }
    .filter(de => !namesToSkip.contains(de.name)).toArray ++ addtlEntries

  val testSetup = reformatMultiLineString(
    """
    .. testsetup::

        vds = hc.read("data/example.vds").annotate_variants_expr('va.genes = ["ACBD", "DCBA"]')
    """)

  def writeFile(filename: String, text: String) = {
    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(text)
    bw.close()
  }

  def reformatMultiLineString(input: String): String = {
    if (input == "")
      input
    else {
      var split = input.split("\n")

      // remove empty header and trailing lines
      if (split.head == "")
        split = split.tail
      if (split.last.matches("\\s*")) {
        split = split.dropRight(1)
      }

      val nCharTrim = split.foldLeft(0) { case (minValue, line) =>
        val trimLength = line.length - line.replaceAll("^\\s+", "").length
        if (minValue == 0)
          trimLength
        else if (trimLength != 0 && trimLength < minValue)
          trimLength
        else
          minValue
      }

      split.map { line =>
        if (line.length < nCharTrim)
          line.replaceAll("^\\s+", "")
        else
          line.substring(math.max(0, nCharTrim))
      }.mkString("\n")
    }
  }

  def makeTypesDocs(filename: String) = {
    val sb = new StringBuilder()

    sb.append(".. sec-types:\n\n")
    sb.append(testSetup)
    sb.append("\n\n")
    sb.append(header1("Types"))
    sb.append("\n\n")

    val types = entries
      .filter(de => de.isField || de.isMethod)
      .groupBy(_.objCategory).toArray.sortBy(_._1)
      .map { case (cat, catEntries) => (cat, catEntries.groupBy(_.objTypePretty)
        .toArray
        .sortBy(_._1).map { case (typ, typEntries) => (typ, typEntries.sortBy(_.name)) })
      }

    // hack to add Struct to documentation
    val typesWStruct = (types ++ Array((Option("Struct"), Array.empty[(String, Array[DocumentationEntry])]))).sortBy(_._1)

    val typeDesc = entries.filter(de => de.isField || de.isMethod)
      .flatMap { de => (de.objCategory, de.objType) match {
        case (Some(n), Some(t)) =>
          Some((n, reformatMultiLineString(t.desc)))
        case (_, _) => None
      }
      }.toMap + ("Struct" -> reformatMultiLineString(TStruct().desc))

    typesWStruct.foreach { case (cat, catEntries) =>
      cat.foreach { c =>
        sb.append(sectionReference(c))
        sb.append(header2(c))
        sb.append("\n\n")
        sb.append(typeDesc(c))
        sb.append("\n\n")
      }
      catEntries.foreach { case (typ, typEntries) =>
        if (catEntries.length != 1) {
          sb.append(sectionReference(typ))
          sb.append(header3(typ))
        }
        typEntries.foreach { de => sb.append(de.emit) }
      }
    }

    writeFile(filename, sb.result())
  }

  def makeFunctionsDocs(filename: String) = {
    val sb = new StringBuilder()

    sb.append(".. sec-functions:\n\n")
    sb.append(testSetup)
    sb.append("\n\n")
    sb.append(header1("Functions"))
    sb.append("\n\n")

    val functions = entries.filter(de => de.isFunction).sortBy(_.name.toLowerCase)
    functions.foreach { de => sb.append(de.emit) }

    writeFile(filename, sb.result())
  }
}
