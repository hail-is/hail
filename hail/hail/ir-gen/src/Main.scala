import scala.annotation.nowarn
import scala.collection.compat._

import mainargs.{main, ParserForMethods}

trait IRDSL {
  // Phantom types used to tag a ConstrainedType
  type Attribute
  type Binding
  type Child
  type Name <: Attribute

  // ConstrainedType

  type Type[+T]

  def att(typ: String): Type[Attribute]
  def binding: Type[Binding]

  def tup[T1, T2](x1: Type[T1], x2: Type[T2]): Type[(T1, T2)]

  def tup[T1, T2, T3](x1: Type[T1], x2: Type[T2], x3: Type[T3]): Type[(T1, T2, T3)]

  protected def zeroOrMore[T](x: Type[T]): Type[Seq[T]]
  protected def oneOrMore[T](x: Type[T]): Type[Seq[T]]
  protected def optional[T](x: Type[T]): Type[Option[T]]

  def name: Type[Name] = att("Name").asInstanceOf[Type[Name]]
  def child: Type[Child]
  def tableChild: Type[Child]
  def matrixChild: Type[Child]
  def bmChild: Type[Child]

  implicit final class TypeOps[+T](x: Type[T]) {
    def * : Type[Seq[T]] = zeroOrMore(x)
    def + : Type[Seq[T]] = oneOrMore(x)
    def ? : Type[Option[T]] = optional(x)
  }

  // Declaration

  type GenericDeclaration
  type Declaration[T] <: DeclarationInterface[T] with GenericDeclaration

  def in[T](name: String, pack: Type[T]): Declaration[T]

  trait DeclarationInterface[T] {
    def withConstraint(c: String => String): Declaration[T]
    def withDefault(value: String): Declaration[T]
    def mutable: Declaration[T]
  }

  // NodeDef

  type NodeDef <: NodeDefInterface

  def node(name: String, attsAndChildren: GenericDeclaration*): NodeDef

  trait NodeDefInterface {
    def withTraits(newTraits: Trait*): NodeDef
    def withPreamble(methodDef: String): NodeDef
    def typed(typ: String): NodeDef
    def withConstraint(c: String): NodeDef
    def withCompanionExtension: NodeDef
    def withClassExtension: NodeDef
    def withDocstring(docstring: String): NodeDef

    def generateDef: String
  }

  // Traits

  type Trait

  val TrivialIR: Trait
  val BaseRef: Trait
  def TypedIR(t: String): Trait
  val NDArrayIR: Trait
  // AbstractApplyNodeUnseededMissingness{Aware, Oblivious}JVMFunction
  def ApplyNode(missingnessAware: Boolean = false): Trait

  // Implicits for common names

  implicit def nameDefaultName(t: Type[Name]): Declaration[Name] =
    in("name", t)

  implicit def childDefaultName(t: Type[Child]): Declaration[Child] =
    in("child", t)
}

object IRDSL_Impl extends IRDSL {
  override def node(name: String, attsAndChildren: Declaration[_]*): NodeDef =
    NodeDef(name, attsAndChildren)

  override def att(typ: String): Type[Attribute] = Att_(typ)
  override val binding: Type[Binding] = Binding

  override def tup[T1, T2](x1: Type[T1], x2: Type[T2]): Type[(T1, T2)] =
    Tup2(x1, x2)

  override def tup[T1, T2, T3](
    x1: Type[T1],
    x2: Type[T2],
    x3: Type[T3],
  ): Type[(T1, T2, T3)] =
    Tup3(x1, x2, x3)

  override def child: Type[Child] = Child_()
  override def tableChild: Type[Child] = Child_("TableIR")
  override def matrixChild: Type[Child] = Child_("MatrixIR")
  override def bmChild: Type[Child] = Child_("BlockMatrixIR")

  override def in[T](name: String, pack: Type[T]): Declaration[T] =
    Declaration(name, pack)

  def int(i: Int): IntRepr = IntRepr(i)

  final case class Trait(name: String)

  override val TrivialIR: Trait = Trait("TrivialIR")
  override val BaseRef: Trait = Trait("BaseRef")
  override def TypedIR(typ: String): Trait = Trait(s"TypedIR[$typ]")
  override val NDArrayIR: Trait = Trait("NDArrayIR")

  override def ApplyNode(missingnessAware: Boolean = false): Trait = {
    val t =
      s"AbstractApplyNode[UnseededMissingness${if (missingnessAware) "Aware" else "Oblivious"}JVMFunction]"
    Trait(t)
  }

  trait Repr[+T] {
    def typ: Type[T]
    def nChildren: IntRepr = int(0)
    def childrenSeq: SeqRepr[Child] = SeqRepr.empty(Child_("BaseIR"))
    def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[T]
  }

  case class IntRepr(static: Int = 0, dynamic: String = "") extends Repr[Int] {
    override def typ: Type[Int] = Att_[Int]("Int")

    override def toString: String = (static, dynamic) match {
      case (s, "") => s.toString
      case (0, d) => d
      case _ => s"$dynamic + $static"
    }

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[Int] = {
      assert(newChildren.hasStaticLen(0))
      this
    }

    def getStatic: Option[Int] = if (dynamic.isEmpty) Some(static) else None

    def hasStaticValue(i: Int): Boolean = getStatic.contains(i)

    def +(other: IntRepr): IntRepr = IntRepr(
      static = static + other.static,
      dynamic = (dynamic, other.dynamic) match {
        case ("", r) => r
        case (l, "") => l
        case (l, r) => s"$l + $r"
      },
    )

    def *(other: IntRepr): IntRepr = (this.getStatic, other.getStatic) match {
      case (Some(0), _) => 0
      case (_, Some(0)) => 0
      case (Some(1), _) => other
      case (_, Some(1)) => this
      case (Some(l), Some(r)) => l * r
      case _ => IntRepr(dynamic = s"($this) * ($other)")
    }
  }

  object IntRepr {
    implicit def makeStatic(i: Int): IntRepr = IntRepr(static = i)
  }

  final case class AttRepr(typ: Att_[Attribute], dynamic: String) extends Repr[Attribute] {
    override def toString: String = dynamic

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[Attribute] = {
      assert(newChildren.hasStaticLen(0))
      this
    }
  }

  final case class ChildRepr(override val typ: Child_, self: String) extends Repr[Child] {
    override def nChildren: IntRepr = 1
    override def childrenSeq: SeqRepr[Child] = SeqRepr.Static(Seq(this), Child_("BaseIR"))
    override def toString: String = self

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[Child] = {
      assert(newChildren.hasStaticLen(1))
      ChildRepr(typ, s"${newChildren(0)}.asInstanceOf[$typ]")
    }
  }

  final case class BindingRepr(self: String) extends Repr[Binding] {
    override def typ: Type[Binding] = Binding
    override def nChildren: IntRepr = 1

    override def childrenSeq: SeqRepr[Child] =
      SeqRepr.Static(Seq(ChildRepr(Child_("BaseIR"), s"$self.value")), Child_("BaseIR"))

    override def toString: String = self

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[Binding] = {
      assert(newChildren.hasStaticLen(1))
      BindingRepr(s"$self.copy(value = ${newChildren(0)}.asInstanceOf[IR])")
    }
  }

  final case class OptRepr[T](eltType: Type[T], self: String) extends Repr[Option[T]] {
    override def typ: Type[Option[T]] = Optional(eltType)
    override def toString: String = self

    override def nChildren: IntRepr = {
      val body = eltType.repr("x").nChildren
      body.getStatic match {
        case Some(0) => 0
        case _ => IntRepr(dynamic = s"$self.map(x => $body).sum")
      }
    }

    override def childrenSeq: SeqRepr[Child] = {
      val body = eltType.repr("x").childrenSeq
      if (body.hasStaticLen(0)) SeqRepr.empty(Child_("BaseIR"))
      else if (eltType.isChild) SeqRepr.Dynamic(s"$self.toSeq", Child_("BaseIR"))
      else
        SeqRepr.Dynamic(s"$self.toSeq.flatMap(x => $body)", Child_("BaseIR"))
    }

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[Option[T]] =
      OptRepr(
        eltType,
        s"$self.map(x => ${eltType.repr("x").copyWithNewChildren(newChildren.slice(0, 1))})",
      )
  }

  sealed abstract class SeqRepr[T] extends Repr[Seq[T]] {
    def eltType: Type[T]
    override def typ: Collection[T] = Collection(eltType)

    def ++(other: SeqRepr[T]): SeqRepr[T] = {
      (this, other) match {
        case (SeqRepr.Static(Seq(), _), r) => r
        case (l, SeqRepr.Static(Seq(), _)) => l
        case (SeqRepr.Static(l, t), SeqRepr.Static(r, _)) => SeqRepr.Static(l ++ r, t)
        case _ => SeqRepr.Dynamic(s"$this ++ $other", eltType)
      }
    }

    def length: IntRepr

    def slice(relStart: IntRepr, len: IntRepr): SeqRepr[T]

    def hasStaticLen(l: Int): Boolean
    def apply(i: IntRepr): Repr[T]

    override def nChildren: IntRepr = {
      val body = typ.elt.repr("x").nChildren
      body.getStatic match {
        case Some(n) => length * n
        case _ => IntRepr(dynamic = s"$this.map(x => $body).sum")
      }
    }

    override def childrenSeq: SeqRepr[Child] =
      if (eltType.repr("x").nChildren.hasStaticValue(0)) SeqRepr.empty(Child_("BaseIR"))
      else if (eltType.isChild) this.asInstanceOf[SeqRepr[Child]]
      else if (eltType.repr("x").nChildren.hasStaticValue(1))
        SeqRepr.Dynamic(s"$this.map(x => ${eltType.repr("x").childrenSeq(0)})", Child_("BaseIR"))
      else
        SeqRepr.Dynamic(s"$this.flatMap(x => ${eltType.repr("x").childrenSeq})", Child_("BaseIR"))

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): SeqRepr[T] = {
      val n = eltType.repr("x").nChildren
      if (n.hasStaticValue(0)) this
      else if (eltType.isChild)
        SeqRepr.Dynamic(s"$newChildren.asInstanceOf[${typ.generateDeclaration}]", eltType)
      else {
        assert(n.hasStaticValue(1))
        SeqRepr.Dynamic(
          s"$this.lazyZip($newChildren).map { (x, newChild) => ${eltType.repr("x").copyWithNewChildren(
              SeqRepr.Static(Seq(ChildRepr(Child_("BaseIR"), "newChild")), Att_("BaseIR"))
            )} }",
          eltType,
        )
      }
    }
  }

  object SeqRepr {
    def empty[T](t: Type[T]): Static[T] = Static[T](Seq.empty, t)

    trait Unsliced[T] extends SeqRepr[T]

    final case class Static[T](elts: Seq[Repr[T]], eltType: Type[T]) extends Unsliced[T] {
      override def typ: Collection[T] = Collection(eltType)
      override def toString: String = s"FastSeq(${elts.mkString(", ")})"
      override def length: IntRepr = elts.length
      override def hasStaticLen(l: Int): Boolean = l == elts.length

      override def slice(relStart: IntRepr, len: IntRepr): SeqRepr[T] =
        (relStart.getStatic, len.getStatic) match {
          case (Some(s), Some(l)) => Static(elts.slice(s, s + l), eltType)
          case (Some(s), None) => Range[T](Static(elts.drop(s), eltType), int(0), len)
          case _ => Range[T](this, relStart, len)
        }

      override def apply(i: IntRepr): Repr[T] = i.getStatic match {
        case Some(i) => elts(i)
        case _ => eltType.repr(s"$this($i)")
      }

      override def nChildren: IntRepr = elts.foldLeft(IntRepr()) { case (acc, elt) =>
        acc + elt.nChildren
      }
    }

    final case class Dynamic[T](elts: String, eltType: Type[T]) extends Unsliced[T] {
      override def typ: Collection[T] = Collection(eltType)
      override def toString: String = elts
      override def length: IntRepr = IntRepr(dynamic = s"$this.length")
      override def hasStaticLen(l: Int): Boolean = false

      override def slice(relStart: IntRepr, len: IntRepr): SeqRepr[T] =
        Range[T](this, relStart, len)

      override def apply(i: IntRepr): Repr[T] = eltType.repr(s"$this($i)")
    }

    final private case class Range[T](
      seq: Unsliced[T],
      start: IntRepr,
      len: IntRepr,
    ) extends SeqRepr[T] {
      override def eltType: Type[T] = seq.eltType
      override def toString: String = s"$seq.slice($start, ${start + len})"
      override def length: IntRepr = len
      override def hasStaticLen(l: Int): Boolean = len.dynamic == "" && len.static == l

      override def slice(relStart: IntRepr, len: IntRepr): SeqRepr[T] =
        Range(seq, start + relStart, len)

      override def apply(i: IntRepr): Repr[T] = eltType.repr(s"$seq(${start + i})")
    }
  }

  final case class TupRepr[T](typ: Tup[T], self: String) extends Repr[T] {
    override def toString: String = self

    override def nChildren: IntRepr =
      typ.elts.zipWithIndex
        .map { case (elt, i) => elt.repr(s"$self._${i + 1}").nChildren }
        .foldLeft(IntRepr())(_ + _)

    override def childrenSeq: SeqRepr[Child] =
      typ.elts.zipWithIndex
        .map { case (elt, i) => elt.repr(s"$self._${i + 1}").childrenSeq }
        .foldLeft[SeqRepr[Child]](SeqRepr.empty[Child](Child_("BaseIR")))(_ ++ _)

    override def copyWithNewChildren(newChildren: SeqRepr[Child]): Repr[T] = {
      val offsets = typ.elts.zipWithIndex.scanLeft(IntRepr()) { case (acc, (elt, i)) =>
        acc + elt.repr(s"$self._${i + 1}").nChildren
      }
      TupRepr(
        typ,
        typ.elts.zipWithIndex.zip(offsets).map { case ((elt, i), n) =>
          val eltRepr = elt.repr(s"$self._${i + 1}")
          eltRepr.copyWithNewChildren(newChildren.slice(n, eltRepr.nChildren))
        }.mkString("(", ", ", ")"),
      )
    }
  }

  override type GenericDeclaration = Declaration[_]

  final case class Declaration[T](
    name: String,
    pack: Type[T],
    isVar: Boolean = false,
    default: Option[String] = None,
    extraConstraints: Seq[String => String] = Seq.empty,
  ) extends DeclarationInterface[T] {
    def repr: Repr[T] = pack.repr(name)
    override def mutable: Declaration[T] = copy(isVar = true)
    override def withDefault(value: String): Declaration[T] = copy(default = Some(value))

    override def withConstraint(c: String => String): Declaration[T] =
      copy(extraConstraints = extraConstraints :+ c)

    def generateDeclaration: String =
      s"${if (isVar) "var " else ""}$name: ${pack.generateDeclaration}${default.map(d => s" = $d").getOrElse("")}"

    def constraints: Seq[String] = (pack.constraints ++ extraConstraints).map(_(name))
    def nChildren: IntRepr = pack.repr(name).nChildren
    def childrenSeq: SeqRepr[Child] = pack.repr(name).childrenSeq
  }

  override def zeroOrMore[T](x: Type[T]): Type[Seq[T]] = Collection(x)
  override def oneOrMore[T](x: Type[T]): Type[Seq[T]] = Collection(x, allowEmpty = false)
  override def optional[T](x: Type[T]): Type[Option[T]] = Optional(x)

  sealed abstract class Type[+T] {
    def isChild: Boolean = false
    def repr(self: String): Repr[T]

    def generateDeclaration: String
    def constraints: Seq[String => String] = Seq.empty
  }

  final case class Att_[T](typ: String) extends Type[T] {
    override def generateDeclaration: String = typ

    override def repr(self: String): Repr[T] =
      if (typ == "Int") IntRepr(dynamic = self).asInstanceOf[Repr[T]]
      else AttRepr(this.asInstanceOf[Att_[Attribute]], self).asInstanceOf[Repr[T]]
  }

  final case class Child_(t: String = "IR") extends Type[Child] {
    override def toString: String = t
    override def isChild: Boolean = true
    override def generateDeclaration: String = t
    override def repr(self: String): Repr[Child] = ChildRepr(this, self)
  }

  case object Binding extends Type[Binding] {
    override def generateDeclaration: String = "Binding"
    override def repr(self: String): Repr[Binding] = BindingRepr(self)
  }

  final case class Optional[T](elt: Type[T]) extends Type[Option[T]] {
    override def generateDeclaration: String = s"Option[${elt.generateDeclaration}]"
    override def repr(self: String): Repr[Option[T]] = OptRepr(elt, self)
  }

  final case class Collection[T](elt: Type[T], allowEmpty: Boolean = true) extends Type[Seq[T]] {
    override def generateDeclaration: String = s"IndexedSeq[${elt.generateDeclaration}]"
    override def repr(self: String): Repr[Seq[T]] = SeqRepr.Dynamic(self, elt)

    override def constraints: Seq[String => String] = {
      val nestedConstraints: Seq[String => String] = if (elt.constraints.nonEmpty)
        Seq(elts =>
          s"$elts.forall(x => ${elt.constraints.map(c => s"(${c("x")})").mkString(" && ")})"
        )
      else Seq()
      val nonEmptyConstraint: Seq[String => String] =
        if (allowEmpty) Seq() else Seq(x => s"$x.nonEmpty")
      nestedConstraints ++ nonEmptyConstraint
    }
  }

  abstract class Tup[T](val elts: Type[_]*) extends Type[T] {
    override def generateDeclaration: String =
      elts.map(_.generateDeclaration).mkString("(", ", ", ")")

    override def constraints: Seq[String => String] =
      elts.zipWithIndex.flatMap { case (elt, i) =>
        elt.constraints.map(c => (t: String) => c(s"$t._${i + 1}"))
      }

    override def repr(self: String): Repr[T] = TupRepr(this, self)
  }

  case class Tup2[T1, T2](x1: Type[T1], x2: Type[T2]) extends Tup[(T1, T2)](x1, x2)

  case class Tup3[T1, T2, T3](
    x1: Type[T1],
    x2: Type[T2],
    x3: Type[T3],
  ) extends Tup[(T1, T2, T3)](x1, x2, x3)

  case class NodeDef(
    name: String,
    attsAndChildren: Seq[Declaration[_]],
    traits: Seq[Trait] = Seq.empty,
    constraints: Seq[String] = Seq.empty,
    extraMethods: Seq[String] = Seq.empty,
    staticMethods: Seq[String] = Seq.empty,
    docstring: String = "",
    hasCompanionExtension: Boolean = false,
  ) extends NodeDefInterface {
    override def withTraits(newTraits: Trait*): NodeDef = copy(traits = traits ++ newTraits)

    override def withPreamble(methodDef: String): NodeDef =
      copy(extraMethods = extraMethods :+ methodDef)

    override def typed(typ: String): NodeDef = withTraits(TypedIR(typ))
    override def withConstraint(c: String): NodeDef = copy(constraints = constraints :+ c)

    override def withCompanionExtension: NodeDef = copy(hasCompanionExtension = true)

    override def withClassExtension: NodeDef = withTraits(Trait(s"${name}Ext"))

    override def withDocstring(docstring: String): NodeDef = copy(docstring = docstring)

    private def nChildren: IntRepr = attsAndChildren.foldLeft(IntRepr())(_ + _.nChildren)

    private def children: String =
      attsAndChildren.foldLeft[SeqRepr[Child]](SeqRepr.empty(Child_("BaseIR")))(
        _ ++ _.childrenSeq
      ).toString

    private def childrenOffsets: Seq[IntRepr] =
      attsAndChildren.scanLeft(IntRepr())(_ + _.nChildren)

    private def copyMethod: String = {
      val decl = s"override def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): $name = "
      val assertion = s"assert(newChildren.length == $nChildren)"
      val body = name + attsAndChildren.lazyZip(childrenOffsets).map { case (x, offset) =>
        val newChildren = SeqRepr.Dynamic("newChildren", Child_("BaseIR"))
        x.repr.copyWithNewChildren(
          if (x.nChildren.hasStaticValue(0)) SeqRepr.empty(Child_("BaseIR"))
          else if (x.nChildren.hasStaticValue(1))
            SeqRepr.Static(Seq(newChildren(offset)), Child_("BaseIR"))
          else if (x.nChildren == nChildren) newChildren
          else newChildren.slice(offset, x.nChildren)
        )
      }.mkString("(", ", ", ")")
      decl + "{\n    " + assertion + "\n    " + body + "}"
    }

    private def classDecl = {
      val paramList = s"$name(${attsAndChildren.map(_.generateDeclaration).mkString(", ")})"
      s"final case class $paramList extends IR" + traits.map(" with " + _.name).mkString
    }

    private def classBody = {
      val childrenSeqDef = if (nChildren.hasStaticValue(0))
        s"override def childrenSeq: IndexedSeq[BaseIR] = IndexedSeq.empty"
      else
        s"override lazy val childrenSeq: IndexedSeq[BaseIR] = $children"
      val extraMethods =
        this.extraMethods :+ childrenSeqDef :+ copyMethod
      val constraints = this.constraints ++ attsAndChildren.flatMap(_.constraints)
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
      (if (docstring.nonEmpty) s"\n/** $docstring*/\n" else "") + classDecl + classBody

    private def companionDef = if (hasCompanionExtension)
      s"object $name extends ${name}CompanionExt\n"
    else ""

    def generateDef: String = companionDef + classDef + "\n"
  }
}

object Main {
  val irdsl: IRDSL = IRDSL_Impl
  import irdsl._

  private val errorID = in("errorID", att("Int")).withDefault("ErrorIDs.NO_ERROR")

  private def _typ(t: String = "Type") = in("_typ", att(t))

  private val mmPerElt =
    in("requiresMemoryManagementPerElement", att("Boolean")).withDefault("false")

  private def allNodes: Seq[NodeDef] = {

    val r = Seq.newBuilder[NodeDef]

    r += node("I32", in("x", att("Int"))).withTraits(TrivialIR)
    r += node("I64", in("x", att("Long"))).withTraits(TrivialIR)
    r += node("F32", in("x", att("Float"))).withTraits(TrivialIR)
    r += node("F64", in("x", att("Double"))).withTraits(TrivialIR)
    r += node("Str", in("x", att("String"))).withTraits(TrivialIR)
      .withPreamble(
        "override def toString(): String = s\"\"\"Str(\"${StringEscapeUtils.escapeString(x)}\")\"\"\""
      ): @nowarn("msg=possible missing interpolator")
    r += node("True").withTraits(TrivialIR)
    r += node("False").withTraits(TrivialIR)
    r += node("Void").withTraits(TrivialIR)
    r += node("NA", _typ()).withTraits(TrivialIR)
    r += node("UUID4", in("id", att("String")))
      .withDocstring(
        """WARNING! This node can only be used when trying to append a one-off,
          |random string that will not be reused elsewhere in the pipeline.
          |Any other uses will need to write and then read again; this node is non-deterministic
          |and will not e.g. exhibit the correct semantics when self-joining on streams.
          |""".stripMargin
      )
      .withCompanionExtension

    r += node(
      "Literal",
      in("_typ", att("Type")).withConstraint(self => s"!CanEmit($self)"),
      in("value", att("Annotation")).withConstraint(self => s"$self != null"),
    )
      .withCompanionExtension
      .withPreamble(
        """// expensive, for debugging
          |// require(SafeRow.isSafe(value))
          |// assert(_typ.typeCheck(value), s"literal invalid:\n  ${_typ}\n  $value")
          |""".stripMargin
      )

    r += node(
      "EncodedLiteral",
      in(
        "codec",
        att("AbstractTypedCodecSpec"),
      ).withConstraint(self =>
        s"!CanEmit($self.encodedVirtualType)"
      ),
      in("value", att("WrappedByteArrays")).withConstraint(self => s"$self != null"),
    )
      .withCompanionExtension

    r += node("Cast", in("v", child), _typ())
    r += node("CastRename", in("v", child), _typ())

    r += node("IsNA", in("value", child))
    r += node("Coalesce", in("values", child.+))
    r += node("Consume", in("value", child))

    r += node("If", in("cond", child), in("cnsq", child), in("altr", child))
    r += node("Switch", in("x", child), in("default", child), in("cases", child.*))
      .withPreamble("override lazy val size: Int = 2 + cases.length")

    r += node("Block", in("bindings", binding.*), in("body", child))
      .withPreamble("override lazy val size: Int = bindings.length + 1")
      .withCompanionExtension

    r += node("Ref", name, _typ().mutable).withTraits(BaseRef)

    r += node(
      "TailLoop",
      name,
      in("params", tup(name, child).*),
      in("resultType", att("Type")),
      in("body", child),
    )
      .withDocstring(
        """Recur can't exist outside of loop. Loops can be nested, but we can't call outer
          |loops in terms of inner loops so there can only be one loop "active" in a given
          |context.
          |""".stripMargin
      )
      .withPreamble("lazy val paramIdx: Map[Name, Int] = params.map(_._1).zipWithIndex.toMap")
    r += node("Recur", name, in("args", child.*), _typ().mutable).withTraits(BaseRef)

    r += node("RelationalLet", name, in("value", child), in("body", child))
    r += node("RelationalRef", name, _typ()).withTraits(BaseRef)

    r += node("ApplyBinaryPrimOp", in("op", att("BinaryOp")), in("l", child), in("r", child))
    r += node("ApplyUnaryPrimOp", in("op", att("UnaryOp")), in("x", child))
    r += node(
      "ApplyComparisonOp",
      in("op", att("ComparisonOp[_]")).mutable,
      in("l", child),
      in("r", child),
    )

    r += node("MakeArray", in("args", child.*), _typ("TArray")).withCompanionExtension
    r += node("MakeStream", in("args", child.*), _typ("TStream"), mmPerElt).withCompanionExtension
    r += node("ArrayRef", in("a", child), in("i", child), errorID)
    r += node(
      "ArraySlice",
      in("a", child),
      in("start", child),
      in("stop", child.?),
      in("step", child).withDefault("I32(1)"),
      errorID,
    )
    r += node("ArrayLen", in("a", child))
    r += node("ArrayZeros", in("length", child))
    r += node(
      "ArrayMaximalIndependentSet",
      in("edges", child),
      in("tieBreaker", tup(name, name, child).?),
    )

    r += node("StreamIota", in("start", child), in("step", child), mmPerElt)
      .withDocstring(
        """[[StreamIota]] is an infinite stream producer, whose element is an integer starting at
          |`start`, updated by `step` at each iteration. The name comes from APL:
          |[[https://stackoverflow.com/questions/9244879/what-does-iota-of-stdiota-stand-for]]
          |""".stripMargin
      )
    r += node(
      "StreamRange",
      in("start", child),
      in("stop", child),
      in("step", child),
      mmPerElt,
      errorID,
    )

    r += node(
      "ArraySort",
      in("a", child),
      in("left", name),
      in("right", name),
      in("lessThan", child),
    )
      .withCompanionExtension

    r += node("ToSet", in("a", child))
    r += node("ToDict", in("a", child))
    r += node("ToArray", in("a", child))
    r += node("CastToArray", in("a", child))
    r += node("ToStream", in("a", child), mmPerElt)
    r += node("GroupByKey", in("collection", child))

    r += node(
      "StreamBufferedAggregate",
      in("streamChild", child),
      in("initAggs", child),
      in("newKey", child),
      in("seqOps", child),
      name,
      in("aggSignature", att("PhysicalAggSig").*),
      in("bufferSize", att("Int")),
    )
    r += node(
      "LowerBoundOnOrderedCollection",
      in("orderedCollection", child),
      in("elem", child),
      in("onKey", att("Boolean")),
    )

    r += node("RNGStateLiteral")
    r += node("RNGSplit", in("state", child), in("dynBitstring", child))

    val key = in("key", att("String").*)

    r += node("StreamLen", in("a", child))
    r += node("StreamGrouped", in("a", child), in("groupSize", child))
    r += node("StreamGroupByKey", in("a", child), key, in("missingEqual", att("Boolean")))
    r += node("StreamMap", in("a", child), name, in("body", child)).typed("TStream")
    r += node("StreamTakeWhile", in("a", child), in("elementName", name), in("body", child))
      .typed("TStream")
    r += node("StreamDropWhile", in("a", child), in("elementName", name), in("body", child))
      .typed("TStream")
    r += node("StreamTake", in("a", child), in("num", child)).typed("TStream")
    r += node("StreamDrop", in("a", child), in("num", child)).typed("TStream")

    r += node(
      "SeqSample",
      in("totalRange", child),
      in("numToSample", child),
      in("rngState", child),
      mmPerElt,
    )
      .typed("TStream")
      .withDocstring(
        """Generate, in ascending order, a uniform random sample, without replacement, of
          |numToSample integers in the range [0, totalRange)
          |""".stripMargin
      )

    r += node(
      "StreamDistribute",
      in("child", child),
      in("pivots", child),
      in("path", child),
      in("comparisonOp", att("ComparisonOp[_]")),
      in("spec", att("AbstractTypedCodecSpec")),
    )
      .withDocstring(
        """Take the child stream and sort each element into buckets based on the provided pivots.
          |The first and last elements of pivots are the endpoints of the first and last interval
          |respectively, should not be contained in the dataset.
          |""".stripMargin
      )

    r += node(
      "StreamWhiten",
      in("stream", child),
      in("newChunk", att("String")),
      in("prevWindow", att("String")),
      in("vecSize", att("Int")),
      in("windowSize", att("Int")),
      in("chunkSize", att("Int")),
      in("blockSize", att("Int")),
      in("normalizeAfterWhiten", att("Boolean")),
    )
      .typed("TStream")
      .withDocstring(
        """"Whiten" a stream of vectors by regressing out from each vector all components
          |in the direction of vectors in the preceding window. For efficiency, takes
          |a stream of "chunks" of vectors.
          |Takes a stream of structs, with two designated fields: `prevWindow` is the
          |previous window (e.g. from the previous partition), if there is one, and
          |`newChunk` is the new chunk to whiten.
          |""".stripMargin
      )

    r += node(
      "StreamZip",
      in("as", child.*),
      in("names", name.*),
      in("body", child),
      in("behavior", att("ArrayZipBehavior.ArrayZipBehavior")),
      errorID,
    )
      .typed("TStream")

    r += node("StreamMultiMerge", in("as", child.*), key).typed("TStream")

    r += node(
      "StreamZipJoinProducers",
      in("contexts", child),
      in("ctxName", name),
      in("makeProducer", child),
      key,
      in("curKey", name),
      in("curVals", name),
      in("joinF", child),
    )
      .typed("TStream")

    r += node(
      "StreamZipJoin",
      in("as", child.*),
      key,
      in("curKey", name),
      in("curVals", name),
      in("joinF", child),
    )
      .typed("TStream")
      .withDocstring(
        """The StreamZipJoin node assumes that input streams have distinct keys. If input streams do not
          |have distinct keys, the key that is included in the result is undefined, but is likely the
          |last.
          |""".stripMargin
      )

    r += node("StreamFilter", in("a", child), name, in("cond", child)).typed("TStream")
    r += node("StreamFlatMap", in("a", child), name, in("cond", child)).typed("TStream")

    r += node(
      "StreamFold",
      in("a", child),
      in("zero", child),
      in("accumName", name),
      in("valueName", name),
      in("body", child),
    )

    r += node(
      "StreamFold2",
      in("a", child),
      in("accum", tup(name, child).*),
      in("valueName", name),
      in("seq", child.*),
      in("result", child),
    )
      .withConstraint("accum.length == seq.length")
      .withPreamble("val nameIdx: Map[Name, Int] = accum.map(_._1).zipWithIndex.toMap")
      .withCompanionExtension

    r += node(
      "StreamScan",
      in("a", child),
      in("zero", child),
      in("accumName", name),
      in("valueName", name),
      in("body", child),
    )
      .typed("TStream")

    r += node("StreamFor", in("a", child), in("valueName", name), in("body", child)).typed(
      "TVoid.type"
    )
    r += node("StreamAgg", in("a", child), name, in("query", child))
    r += node("StreamAggScan", in("a", child), name, in("query", child)).typed("TStream")

    r += node(
      "StreamLeftIntervalJoin",
      in("left", child),
      in("right", child),
      in("lKeyFieldName", att("String")),
      in("rIntervalFieldName", att("String")),
      in("lname", name),
      in("rname", name),
      in("body", child),
    )
      .typed("TStream")

    r += node(
      "StreamJoinRightDistinct",
      in("left", child),
      in("right", child),
      in("lKey", att("String").*),
      in("rKey", att("String").*),
      in("l", name),
      in("r", name),
      in("joinF", child),
      in("joinType", att("String")),
    )
      .typed("TStream").withClassExtension

    r += node(
      "StreamLocalLDPrune",
      in("child", child),
      in("r2Threshold", child),
      in("windowSize", child),
      in("maxQueueSize", child),
      in("nSamples", child),
    )
      .typed("TStream")

    r += node("MakeNDArray", in("data", child), in("shape", child), in("rowMajor", child), errorID)
      .withTraits(NDArrayIR).withCompanionExtension
    r += node("NDArrayShape", in("nd", child))
    r += node("NDArrayReshape", in("nd", child), in("shape", child), errorID).withTraits(NDArrayIR)
    r += node("NDArrayConcat", in("nds", child), in("axis", att("Int"))).withTraits(NDArrayIR)
    r += node("NDArrayRef", in("nd", child), in("idxs", child.*), errorID)
    r += node("NDArraySlice", in("nd", child), in("slices", child)).withTraits(NDArrayIR)
    r += node("NDArrayFilter", in("nd", child), in("keep", child.*)).withTraits(NDArrayIR)
    r += node("NDArrayMap", in("nd", child), in("valueName", name), in("body", child))
      .withTraits(NDArrayIR)
    r += node(
      "NDArrayMap2",
      in("l", child),
      in("r", child),
      in("lName", name),
      in("rName", name),
      in("body", child),
      errorID,
    )
      .withTraits(NDArrayIR)
    r += node("NDArrayReindex", in("nd", child), in("indexExpr", att("Int").*))
      .withTraits(NDArrayIR)
    r += node("NDArrayAgg", in("nd", child), in("axes", att("Int").*))
    r += node("NDArrayWrite", in("nd", child), in("path", child)).typed("TVoid.type")
    r += node("NDArrayMatMul", in("l", child), in("r", child), errorID).withTraits(NDArrayIR)
    r += node(
      "NDArrayQR",
      in("nd", child),
      in("mode", att("String")),
      errorID,
    ).withCompanionExtension
    r += node(
      "NDArraySVD",
      in("nd", child),
      in("fullMatrices", att("Boolean")),
      in("computeUV", att("Boolean")),
      errorID,
    )
      .withCompanionExtension
    r += node("NDArrayEigh", in("nd", child), in("eigvalsOnly", att("Boolean")), errorID)
      .withCompanionExtension
    r += node("NDArrayInv", in("nd", child), errorID).withTraits(NDArrayIR).withCompanionExtension

    val isScan = in("isScan", att("Boolean"))

    r += node("AggFilter", in("cond", child), in("aggIR", child), isScan)
    r += node("AggExplode", in("array", child), name, in("aggBody", child), isScan)
    r += node("AggGroupBy", in("key", child), in("aggIR", child), isScan)
    r += node(
      "AggArrayPerElement",
      in("a", child),
      in("elementName", name),
      in("indexName", name),
      in("aggBody", child),
      in("knownLength", child.?),
      isScan,
    )
    r += node(
      "AggFold",
      in("zero", child),
      in("seqOp", child),
      in("combOp", child),
      in("accumName", name),
      in("otherAccumName", name),
      isScan,
    )
      .withCompanionExtension

    r += node(
      "ApplyAggOp",
      in("initOpArgs", child.*),
      in("seqOpArgs", child.*),
      in("op", att("AggOp")),
    )
      .withCompanionExtension
    r += node(
      "ApplyScanOp",
      in("initOpArgs", child.*),
      in("seqOpArgs", child.*),
      in("op", att("AggOp")),
    )
      .withClassExtension.withCompanionExtension
    r += node(
      "InitOp",
      in("i", att("Int")),
      in("args", child.*),
      in("aggSig", att("PhysicalAggSig")),
    )
    r += node(
      "SeqOp",
      in("i", att("Int")),
      in("args", child.*),
      in("aggSig", att("PhysicalAggSig")),
    )
    r += node(
      "CombOp",
      in("i1", att("Int")),
      in("i2", att("Int")),
      in("aggSig", att("PhysicalAggSig")),
    )
    r += node("ResultOp", in("idx", att("Int")), in("aggSig", att("PhysicalAggSig")))
      .withCompanionExtension
    r += node(
      "CombOpValue",
      in("i", att("Int")),
      in("value", child),
      in("aggSig", att("PhysicalAggSig")),
    )
    r += node("AggStateValue", in("i", att("Int")), in("aggSig", att("AggStateSig")))
    r += node(
      "InitFromSerializedValue",
      in("i", att("Int")),
      in("value", child),
      in("aggSig", att("AggStateSig")),
    )
    r += node(
      "SerializeAggs",
      in("startIdx", att("Int")),
      in("serializedIdx", att("Int")),
      in("spec", att("BufferSpec")),
      in("aggSigs", att("AggStateSig").*),
    )
    r += node(
      "DeserializeAggs",
      in("startIdx", att("Int")),
      in("serializedIdx", att("Int")),
      in("spec", att("BufferSpec")),
      in("aggSigs", att("AggStateSig").*),
    )
    r += node(
      "RunAgg",
      in("body", child),
      in("result", child),
      in("signature", att("AggStateSig").*),
    )
    r += node(
      "RunAggScan",
      in("array", child),
      name,
      in("init", child),
      in("seqs", child),
      in("result", child),
      in("signature", att("AggStateSig").*),
    )

    r += node("MakeStruct", in("fields", tup(att("String"), child).*)).typed("TStruct")
    r += node("SelectFields", in("old", child), in("fields", att("String").*))
      .typed("TStruct")
    r += node(
      "InsertFields",
      in("old", child),
      in("fields", tup(att("String"), child).*),
      in("fieldOrder", att("String").*.?).withDefault("None"),
    )
      .typed("TStruct")
    r += node("GetField", in("o", child), in("name", att("String")))
    r += node("MakeTuple", in("fields", tup(att("Int"), child).*))
      .typed("TTuple").withCompanionExtension
    r += node("GetTupleElement", in("o", child), in("idx", att("Int")))

    r += node("In", in("i", att("Int")), in("_typ", att("EmitParamType")))
      .withDocstring("Function input").withCompanionExtension

    r += node("Die", in("message", child), in("_typ", att("Type")), errorID).withCompanionExtension
    r += node("Trap", in("child", child)).withDocstring(
      """The Trap node runs the `child` node with an exception handler. If the child throws a
        |HailException (user exception), then we return the tuple ((msg, errorId), NA). If the child
        |throws any other exception, we raise that exception. If the child does not throw, then we
        |return the tuple (NA, child value).
        |""".stripMargin
    )
    r += node("ConsoleLog", in("message", child), in("result", child))

    r += node(
      "ApplyIR",
      in("function", att("String")),
      in("typeArgs", att("Seq[Type]")),
      in("args", child.*),
      in("returnType", att("Type")),
      errorID,
    )
      .withClassExtension

    r += node(
      "Apply",
      in("function", att("String")),
      in("typeArgs", att("Seq[Type]")),
      in("args", child.*),
      in("returnType", att("Type")),
      errorID,
    ).withTraits(ApplyNode())

    r += node(
      "ApplySeeded",
      in("function", att("String")),
      in("_args", child.*),
      in("rngState", child),
      in("staticUID", att("Long")),
      in("returnType", att("Type")),
    ).withTraits(ApplyNode())
      .withPreamble("val args = rngState +: _args")
      .withPreamble("val typeArgs: Seq[Type] = Seq.empty[Type]")

    r += node(
      "ApplySpecial",
      in("function", att("String")),
      in("typeArgs", att("Seq[Type]")),
      in("args", child.*),
      in("returnType", att("Type")),
      errorID,
    ).withTraits(ApplyNode(missingnessAware = true))

    r += node("LiftMeOut", in("child", child))

    r += node("TableCount", tableChild)
    r += node("MatrixCount", matrixChild)
    r += node("TableAggregate", tableChild, in("query", child))
    r += node("MatrixAggregate", matrixChild, in("query", child))
    r += node("TableWrite", tableChild, in("writer", att("TableWriter")))
    r += node(
      "TableMultiWrite",
      in("_children", tableChild.*),
      in("writer", att("WrappedMatrixNativeMultiWriter")),
    )
    r += node("TableGetGlobals", tableChild)
    r += node("TableCollect", tableChild)
    r += node("MatrixWrite", matrixChild, in("writer", att("MatrixWriter")))
    r += node(
      "MatrixMultiWrite",
      in("_children", matrixChild.*),
      in("writer", att("MatrixNativeMultiWriter")),
    )
    r += node("TableToValueApply", tableChild, in("function", att("TableToValueFunction")))
    r += node("MatrixToValueApply", matrixChild, in("function", att("MatrixToValueFunction")))
    r += node(
      "BlockMatrixToValueApply",
      bmChild,
      in("function", att("BlockMatrixToValueFunction")),
    )
    r += node("BlockMatrixCollect", bmChild)
    r += node("BlockMatrixWrite", bmChild, in("writer", att("BlockMatrixWriter")))
    r += node(
      "BlockMatrixMultiWrite",
      in("blockMatrices", bmChild.*),
      in("writer", att("BlockMatrixMultiWriter")),
    )

    r += node(
      "CollectDistributedArray",
      in("contexts", child),
      in("globals", child),
      in("cname", name),
      in("gname", name),
      in("body", child),
      in("dynamicID", child),
      in("staticID", att("String")),
      in("tsd", att("Option[TableStageDependency]")).withDefault("None"),
    )

    r += node(
      "ReadPartition",
      in("context", child),
      in("rowType", att("TStruct")),
      in("reader", att("PartitionReader")),
    )
    r += node(
      "WritePartition",
      in("value", child),
      in("writeCtx", child),
      in("writer", att("PartitionWriter")),
    )
    r += node("WriteMetadata", in("writeAnnotations", child), in("writer", att("MetadataWriter")))
    r += node(
      "ReadValue",
      in("path", child),
      in("reader", att("ValueReader")),
      in("requestedType", att("Type")),
    )
    r += node(
      "WriteValue",
      in("value", child),
      in("path", child),
      in("writer", att("ValueWriter")),
      in("stagingFile", child.?).withDefault("None"),
    )

    r.result()
  }

  @main
  def main(path: String): Unit = {
    val pack = "package is.hail.expr.ir.defs"
    val imports = Seq(
      "is.hail.annotations.Annotation",
      "is.hail.io.{AbstractTypedCodecSpec, BufferSpec}",
      "is.hail.types.virtual.{Type, TArray, TStream, TVoid, TStruct, TTuple}",
      "is.hail.utils.{FastSeq, StringEscapeUtils}",
      "is.hail.expr.ir.{AggOp, BaseIR, IR, TableIR, MatrixIR, BlockMatrixIR, Name, UnaryOp, BinaryOp, " +
        "ComparisonOp, CanEmit, EmitParamType, TableWriter, " +
        "WrappedMatrixNativeMultiWriter, MatrixWriter, MatrixNativeMultiWriter, BlockMatrixWriter, " +
        "BlockMatrixMultiWriter, ValueReader, ValueWriter}",
      "is.hail.expr.ir.lowering.TableStageDependency",
      "is.hail.expr.ir.agg.{PhysicalAggSig, AggStateSig}",
      "is.hail.expr.ir.functions.{UnseededMissingnessAwareJVMFunction, " +
        "UnseededMissingnessObliviousJVMFunction, TableToValueFunction, MatrixToValueFunction, " +
        "BlockMatrixToValueFunction}",
      "is.hail.expr.ir.defs.exts._",
      "scala.collection.compat._",
    )
    val gen = pack + "\n\n" + imports.map(i => s"import $i").mkString("\n") + "\n\n" + allNodes.map(
      _.generateDef
    ).mkString("\n")
    os.write(os.Path(path) / "IR_gen.scala", gen)
  }

  def main(args: Array[String]): Unit = {
    val _ = ParserForMethods(this).runOrExit(args)
  }
}
