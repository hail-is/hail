import mainargs.{ParserForMethods, main}

sealed abstract class Trait(val name: String)

object Trivial extends Trait("TrivialIR")

case class NChildren(static: Int = 0, dynamic: String = "") {
  def +(other: NChildren): NChildren = NChildren(
    static = static + other.static,
    dynamic = if (dynamic.isEmpty) other.dynamic else s"$dynamic + ${other.dynamic}",
  )
}

sealed abstract class AttOrChild {
  val name: String
  def generateDeclaration: String
  def constraints: Seq[String] = Seq.empty
  def nChildren: NChildren = NChildren()
}

final case class Att(name: String, typ: String) extends AttOrChild {
  override def generateDeclaration: String = s"$name: $typ"
}

final case class Child(name: String) extends AttOrChild {
  override def generateDeclaration: String = s"$name: IR"
  override def nChildren: NChildren = NChildren(static = 1)
}

final case class ChildPlus(name: String) extends AttOrChild {
  override def generateDeclaration: String = s"$name: IndexedSeq[IR]"
  override def constraints: Seq[String] = Seq(s"$name.nonEmpty")
  override def nChildren: NChildren = NChildren(dynamic = "name.size")
}

final case class ChildStar(name: String) extends AttOrChild {
  override def generateDeclaration: String = s"$name: IndexedSeq[IR]"
  override def nChildren: NChildren = NChildren(dynamic = "name.size")
}

case class IR(
  name: String,
  attsAndChildren: Seq[AttOrChild],
  traits: Seq[Trait] = Seq.empty,
  extraMethods: Seq[String] = Seq.empty,
  applyMethods: Seq[String] = Seq.empty,
  docstring: String = "",
) {
  def withTraits(newTraits: Trait*): IR = copy(traits = traits ++ newTraits)
  def withMethod(methodDef: String): IR = copy(extraMethods = extraMethods :+ methodDef)
  def withApply(methodDef: String): IR = copy(applyMethods = applyMethods :+ methodDef)
  def withDocstring(docstring: String): IR = copy(docstring = docstring)

  private def nChildren: NChildren = attsAndChildren.foldLeft(NChildren())(_ + _.nChildren)

  private def children: String = {
    val tmp = attsAndChildren.flatMap {
      case _: Att => None
      case c: Child => Some(s"FastSeq(${c.name})")
      case cs: ChildPlus => Some(cs.name)
      case cs: ChildStar => Some(cs.name)
    }
    if (tmp.isEmpty) "FastSeq.empty" else tmp.mkString(" ++ ")
  }

  private def paramList = s"$name(${attsAndChildren.map(_.generateDeclaration).mkString(", ")})"

  private def classDecl =
    s"final case class $paramList extends IR" + traits.map(" with " + _.name).mkString

  private def classBody = {
    val extraMethods =
      this.extraMethods :+ s"override lazy val childrenSeq: IndexedSeq[IR] = $children"
    val constraints = attsAndChildren.flatMap(_.constraints)
    if (constraints.nonEmpty || extraMethods.nonEmpty) {
      (
        " {" +
          (if (constraints.nonEmpty)
             constraints.map(c => s"  require($c)").mkString("\n", "\n", "\n")
           else "")
          + (
            if (extraMethods.nonEmpty)
              extraMethods.map("  " + _).mkString("\n", "\n", "\n")
            else ""
          )
          + "}"
      )
    } else ""
  }

  private def classDef =
    (if (docstring.nonEmpty) s"\n// $docstring\n" else "") + classDecl + classBody

  private def companionBody = applyMethods.map("  " + _).mkString("\n")

  private def companionDef =
    if (companionBody.isEmpty) "" else s"object $name {\n$companionBody\n}\n"

  def generateDef: String = companionDef + classDef + "\n"
}

object Main {
  def node(name: String, attsAndChildren: AttOrChild*): IR = IR(name, attsAndChildren)

  def allNodes: Seq[IR] = {
    val r = Seq.newBuilder[IR]

    r += node("I32", Att("x", "Int")).withTraits(Trivial)
    r += node("I64", Att("x", "Long")).withTraits(Trivial)
    r += node("F32", Att("x", "Float")).withTraits(Trivial)
    r += node("F64", Att("x", "Double")).withTraits(Trivial)
    r += node("Str", Att("x", "String")).withTraits(Trivial)
      .withMethod(
        "override def toString(): String = s\"\"\"Str(\"${StringEscapeUtils.escapeString(x)}\")\"\"\""
      )
    r += node("True").withTraits(Trivial)
    r += node("False").withTraits(Trivial)
    r += node("Void").withTraits(Trivial)
    r += node("NA", Att("_typ", "Type")).withTraits(Trivial)
    r += node("UUID4", Att("id", "String"))
      .withDocstring(
        "WARNING! This node can only be used when trying to append a one-off, "
          + "random string that will not be reused elsewhere in the pipeline. "
          + "Any other uses will need to write and then read again; this node is non-deterministic "
          + "and will not e.g. exhibit the correct semantics when self-joining on streams."
      )
      .withApply("def apply(): UUID4 = UUID4(genUID())")
    r += node("Cast", Child("v"), Att("_typ", "Type"))
    r += node("CastRename", Child("v"), Att("_typ", "Type"))
    r += node("IsNA", Child("value"))
    r += node("Coalesce", ChildPlus("values"))
    r += node("Consume", Child("value"))
    r += node("If", Child("cond"), Child("cnsq"), Child("altr"))
    r += node("Switch", Child("x"), Child("default"), ChildStar("cases"))
      .withMethod("override lazy val size: Int = 2 + cases.length")

    r.result()
  }

  @main
  def main(path: String) = {
    val pack = "package is.hail.expr.ir"
    val imports = Seq("is.hail.types.virtual.Type", "is.hail.utils.{FastSeq, StringEscapeUtils}")
    val gen = pack + "\n\n" + imports.map(i => s"import $i").mkString("\n") + "\n\n" + allNodes.map(
      _.generateDef
    ).mkString("\n")
    os.write(os.Path(path) / "IR_gen.scala", gen)
  }

  def main(args: Array[String]): Unit = ParserForMethods(this).runOrExit(args)
}
