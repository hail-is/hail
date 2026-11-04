import scala.annotation.nowarn
import scala.collection.immutable.ArraySeq

import Family.Family
import mainargs.{main, ParserForMethods}

// The group an IR node's class belongs to. A node's tag holds its family's `id` in the
// bits above `Family.FamilyShift`, and its position within the family below, so a pass
// can switch on the family and then on the tag, and each switch compiles to a jump table.
//
// Relational, Table and Matrix hold the nodes a lowering pass removes: a node goes in the
// family of the first invariant that forbids it. For example, NoMatrixIR forbids every
// node that is or reads a MatrixIR, so LowerMatrixIR removes the whole Matrix family.
// BlockMatrix is the odd one out: CompilableIR forbids it at the same point as Table, so
// a node goes in BlockMatrix if it reads or makes a block matrix but no Table.
//
// The other families are buckets that keep each pass's per-family method small. A node
// goes in the family of what it produces.

object Family extends Enumeration {
  type Family = Value
  final val FamilyShift = 8

  // a constant value
  val Const

  // a ref or input, or binds, branches, loops, throws, calls a function or
  // applies a built-in operator; also one-off helpers (UUID4, Consume,
  // ConsoleLog)
  , Core

  // produces or reads a struct or tuple
  , Struct

  // produces, reads, transforms or writes a stream
  , Stream

  // produces or reads an array, set or dict
  , Collection

  // uses an agg context or agg state
  , Agg

  // reads or writes a partition, value or metadata
  , IO

  // produces, reads, transforms or writes an ndarray
  , NDArray

  // creates or splits RNG state
  , Random

  // binds or refers to a value computed from relational IR
  , Relational

  // reads, transforms or writes a Table
  , Table

  // reads, transforms or writes a MatrixTable
  , Matrix

  // reads, transforms or writes a block matrix
  , BlockMatrix = Value
}

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

  final def name: Type[Name] = att("Name").asInstanceOf[Type[Name]]
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

  def node(family: Family, name: String, attsAndChildren: GenericDeclaration*): NodeDef

  /** A node whose class is written by hand, outside ir-gen. Only its tag is generated. */
  def handWritten(family: Family, name: String): NodeDef

  trait NodeDefInterface {
    def family: Family
    def name: String
    def isHandWritten: Boolean
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

  val Atom: Trait
  val BaseRef: Trait
  def TypedIR(t: String): Trait
  // AbstractApplyNodeUnseededMissingness{Aware, Oblivious}JVMFunction
  def ApplyNode(missingnessAware: Boolean = false): Trait

  // Implicits for common names

  implicit final def nameDefaultName(t: Type[Name]): Declaration[Name] =
    in("name", t)

  implicit final def childDefaultName(t: Type[Child]): Declaration[Child] =
    in("child", t)
}

object IRDSL_Impl extends IRDSL {
  override def node(family: Family, name: String, attsAndChildren: Declaration[_]*): NodeDef =
    NodeDef(family, name, attsAndChildren)

  override def handWritten(family: Family, name: String): NodeDef =
    NodeDef(family, name, Seq.empty, isHandWritten = true)

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

  override val Atom: Trait = Trait("Atom")
  override val BaseRef: Trait = Trait("BaseRef")
  override def TypedIR(typ: String): Trait = Trait(s"TypedIR[$typ]")

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
    family: Family,
    name: String,
    attsAndChildren: Seq[Declaration[_]],
    traits: Seq[Trait] = Seq.empty,
    constraints: Seq[String] = Seq.empty,
    extraMethods: Seq[String] = Seq.empty,
    staticMethods: Seq[String] = Seq.empty,
    docstring: String = "",
    hasCompanionExtension: Boolean = false,
    isHandWritten: Boolean = false,
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
      s"final case class $paramList extends IR(new Tag(Tag.$name))" + traits.map(
          " with " + _.name
        ).mkString
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

    override def generateDef: String = {
      require(!isHandWritten, s"$name is hand-written; only its tag is generated")
      companionDef + classDef + "\n"
    }
  }
}

object Main {
  val irdsl: IRDSL = IRDSL_Impl

  import irdsl._
  import Family._

  private val errorID = in("errorID", att("Int")).withDefault("ErrorIDs.NO_ERROR")
  private val isScan = in("isScan", att("Boolean"))
  private val key = in("key", att("String").*)

  private def _typ(t: String = "Type") = in("_typ", att(t))

  private val mmPerElt =
    in("requiresMemoryManagementPerElement", att("Boolean")).withDefault("false")

  private def allNodes: Seq[NodeDef] = {

    val r = Seq.newBuilder[NodeDef]

    r += node(Const, "NA", _typ()).withTraits(Atom)
    r += node(Const, "I32", in("x", att("Int"))).withTraits(Atom)
    r += node(Const, "I64", in("x", att("Long"))).withTraits(Atom)
    r += node(Const, "F32", in("x", att("Float"))).withTraits(Atom)
    r += node(Const, "F64", in("x", att("Double"))).withTraits(Atom)

    // Making Str < Atom would lead to code bloat
    r += node(Const, "Str", in("x", att("String")))
      .withPreamble(
        "override def toString(): String = s\"\"\"Str(\"${StringEscapeUtils.escapeString(x)}\")\"\"\""
      ): @nowarn("msg=possible missing interpolator")

    r += node(Const, "True").withTraits(Atom)
    r += node(Const, "False").withTraits(Atom)
    r += node(Const, "Void").withTraits(Atom)

    r += node(
      Const,
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
      Const,
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

    r += node(Core, "Cast", in("v", child), _typ())
    r += node(Core, "Coalesce", in("values", child.+))
    r += node(Core, "If", in("cond", child), in("cnsq", child), in("altr", child))

    r += node(Core, "Switch", in("x", child), in("default", child), in("cases", child.*))
      .withPreamble("override lazy val size: Int = 2 + cases.length")

    r += node(Core, "Ref", name, _typ().mutable).withTraits(BaseRef, Atom)
    r += node(Core, "Recur", name, in("args", child.*), _typ().mutable).withTraits(BaseRef)

    r += node(Core, "In", in("i", att("Int")), in("_typ", att("EmitParamType")))
      .withTraits(Atom)
      .withDocstring("Function input")
      .withCompanionExtension

    r += node(
      Core,
      "Die",
      in("message", child),
      in("_typ", att("Type")),
      errorID,
    ).withCompanionExtension

    r += node(Core, "ConsoleLog", in("message", child), in("result", child))

    r += node(
      Core,
      "ApplyBinaryPrimOp",
      in("op", att("BinaryOp")),
      in("l", child),
      in("r", child),
    )

    r += node(Core, "ApplyUnaryPrimOp", in("op", att("UnaryOp")), in("x", child))

    r += node(
      Core,
      "ApplyComparisonOp",
      in("op", att("ComparisonOp[_]")).mutable,
      in("l", child),
      in("r", child),
    )

    r += node(
      Core,
      "Apply",
      in("function", att("String")),
      in("typeArgs", att("IndexedSeq[Type]")),
      in("args", child.*),
      in("returnType", att("Type")),
      errorID,
    ).withTraits(ApplyNode())

    r += node(
      Core,
      "ApplySpecial",
      in("function", att("String")),
      in("typeArgs", att("IndexedSeq[Type]")),
      in("args", child.*),
      in("returnType", att("Type")),
      errorID,
    ).withTraits(ApplyNode(missingnessAware = true))

    r += node(Core, "Block", in("bindings", binding.*), in("body", child))
      .withPreamble("override lazy val size: Int = bindings.length + 1")
      .withCompanionExtension

    r += node(
      Core,
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

    r += node(Core, "UUID4", in("id", att("String")))
      .withDocstring(
        """WARNING! This node can only be used when trying to append a one-off,
          |random string that will not be reused elsewhere in the pipeline.
          |Any other uses will need to write and then read again; this node is non-deterministic
          |and will not e.g. exhibit the correct semantics when self-joining on streams.
          |""".stripMargin
      )
      .withCompanionExtension

    r += node(Core, "IsNA", in("value", child))
    r += node(Core, "Consume", in("value", child))

    r += node(
      Core,
      "ApplyIR",
      in("function", att("String")),
      in("typeArgs", att("IndexedSeq[Type]")),
      in("args", child.*),
      in("returnType", att("Type")),
      errorID,
    )
      .withClassExtension

    r += node(Struct, "CastRename", in("v", child), _typ())
    r += node(Struct, "MakeStruct", in("fields", tup(att("String"), child).*)).typed("TStruct")

    r += node(Struct, "SelectFields", in("old", child), in("fields", att("String").*))
      .typed("TStruct")

    r += node(
      Struct,
      "InsertFields",
      in("old", child),
      in("fields", tup(att("String"), child).*),
      in("fieldOrder", att("String").*.?).withDefault("None"),
    )
      .typed("TStruct")

    r += node(Struct, "GetField", in("o", child), in("name", att("String")))

    r += node(Struct, "MakeTuple", in("fields", tup(att("Int"), child).*))
      .typed("TTuple").withCompanionExtension

    r += node(Struct, "GetTupleElement", in("o", child), in("idx", att("Int")))

    r += node(Stream, "MakeStream", in("args", child.*), _typ("TStream"), mmPerElt)
      .typed("TStream")
      .withCompanionExtension

    r += node(Stream, "StreamIota", in("start", child), in("step", child), mmPerElt)
      .withDocstring(
        """[[StreamIota]] is an infinite stream producer, whose element is an integer starting at
          |`start`, updated by `step` at each iteration. The name comes from APL:
          |[[https://stackoverflow.com/questions/9244879/what-does-iota-of-stdiota-stand-for]]
          |""".stripMargin
      )

    r += node(
      Stream,
      "StreamRange",
      in("start", child),
      in("stop", child),
      in("step", child),
      mmPerElt,
      errorID,
    )

    r += node(Stream, "ToStream", in("a", child), mmPerElt).typed("TStream")
    r += node(Stream, "StreamLen", in("a", child))
    r += node(Stream, "StreamGrouped", in("a", child), in("groupSize", child))
    r += node(Stream, "StreamGroupByKey", in("a", child), key, in("missingEqual", att("Boolean")))
    r += node(Stream, "StreamTake", in("a", child), in("num", child)).typed("TStream")
    r += node(Stream, "StreamDrop", in("a", child), in("num", child)).typed("TStream")

    r += node(
      Stream,
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
      Stream,
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
      Stream,
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

    r += node(Stream, "StreamMultiMerge", in("as", child.*), key).typed("TStream")

    r += node(
      Stream,
      "StreamLocalLDPrune",
      in("child", child),
      in("r2Threshold", child),
      in("windowSize", child),
      in("maxQueueSize", child),
      in("nSamples", child),
    )
      .typed("TStream")

    r += node(
      Stream,
      "StreamBufferedAggregate",
      in("streamChild", child),
      in("initAggs", child),
      in("newKey", child),
      in("seqOps", child),
      name,
      in("aggSignature", att("PhysicalAggSig").*),
      in("bufferSize", att("Int")),
    )

    r += node(Stream, "StreamMap", in("a", child), name, in("body", child)).typed("TStream")

    r += node(Stream, "StreamTakeWhile", in("a", child), in("elementName", name), in("body", child))
      .typed("TStream")

    r += node(Stream, "StreamDropWhile", in("a", child), in("elementName", name), in("body", child))
      .typed("TStream")

    r += node(
      Stream,
      "StreamZip",
      in("as", child.*),
      in("names", name.*),
      in("body", child),
      in("behavior", att("ArrayZipBehavior.ArrayZipBehavior")),
      errorID,
    )
      .typed("TStream")

    r += node(
      Stream,
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
      Stream,
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

    r += node(Stream, "StreamFilter", in("a", child), name, in("cond", child)).typed("TStream")
    r += node(Stream, "StreamFlatMap", in("a", child), name, in("cond", child)).typed("TStream")

    r += node(
      Stream,
      "StreamFold",
      in("a", child),
      in("zero", child),
      in("accumName", name),
      in("valueName", name),
      in("body", child),
    )

    r += node(
      Stream,
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
      Stream,
      "StreamScan",
      in("a", child),
      in("zero", child),
      in("accumName", name),
      in("valueName", name),
      in("body", child),
    )
      .typed("TStream")

    r += node(Stream, "StreamFor", in("a", child), in("valueName", name), in("body", child)).typed(
      "TVoid.type"
    )

    r += node(Stream, "StreamAgg", in("a", child), name, in("query", child))
    r += node(Stream, "StreamAggScan", in("a", child), name, in("query", child)).typed("TStream")

    r += node(
      Stream,
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
      Stream,
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

    r += node(Collection, "MakeArray", in("args", child.*), _typ("TArray")).withCompanionExtension
    r += node(Collection, "ArrayRef", in("a", child), in("i", child), errorID)

    r += node(
      Collection,
      "ArraySlice",
      in("a", child),
      in("start", child),
      in("stop", child.?),
      in("step", child).withDefault("I32(1)"),
      errorID,
    )

    r += node(Collection, "ArrayLen", in("a", child))
    r += node(Collection, "ArrayZeros", in("length", child))
    r += node(Collection, "ToSet", in("a", child))
    r += node(Collection, "ToDict", in("a", child))
    r += node(Collection, "ToArray", in("a", child)).typed("TArray")
    r += node(Collection, "CastToArray", in("a", child)).typed("TArray")
    r += node(Collection, "GroupByKey", in("collection", child))

    r += node(
      Collection,
      "LowerBoundOnOrderedCollection",
      in("orderedCollection", child),
      in("elem", child),
      in("onKey", att("Boolean")),
    )

    r += node(
      Collection,
      "ArrayMaximalIndependentSet",
      in("edges", child),
      in("tieBreaker", tup(name, name, child).?),
    )

    r += node(
      Collection,
      "ArraySort",
      in("a", child),
      in("left", name),
      in("right", name),
      in("lessThan", child),
    )
      .withCompanionExtension

    r += node(
      Collection,
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

    // Agg
    r += node(
      Agg,
      "InitOp",
      in("i", att("Int")),
      in("args", child.*),
      in("aggSig", att("PhysicalAggSig")),
    )

    r += node(
      Agg,
      "SeqOp",
      in("i", att("Int")),
      in("args", child.*),
      in("aggSig", att("PhysicalAggSig")),
    )

    r += node(
      Agg,
      "CombOpValue",
      in("i", att("Int")),
      in("value", child),
      in("aggSig", att("PhysicalAggSig")),
    )

    r += node(
      Agg,
      "InitFromSerializedValue",
      in("i", att("Int")),
      in("value", child),
      in("aggSig", att("AggStateSig")),
    )

    r += node(
      Agg,
      "RunAgg",
      in("body", child),
      in("result", child),
      in("signature", att("AggStateSig").*),
    )

    r += node(Agg, "AggFilter", in("cond", child), in("aggIR", child), isScan)
    r += node(Agg, "AggExplode", in("array", child), name, in("aggBody", child), isScan)
    r += node(Agg, "AggGroupBy", in("key", child), in("aggIR", child), isScan)

    r += node(
      Agg,
      "AggArrayPerElement",
      in("a", child),
      in("elementName", name),
      in("indexName", name),
      in("aggBody", child),
      in("knownLength", child.?),
      isScan,
    )

    r += node(
      Agg,
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
      Agg,
      "ApplyAggOp",
      in("initOpArgs", child.*),
      in("seqOpArgs", child.*),
      in("op", att("AggOp")),
    )
      .withCompanionExtension

    r += node(
      Agg,
      "ApplyScanOp",
      in("initOpArgs", child.*),
      in("seqOpArgs", child.*),
      in("op", att("AggOp")),
    )
      .withClassExtension.withCompanionExtension

    r += node(
      Agg,
      "RunAggScan",
      in("array", child),
      name,
      in("init", child),
      in("seqs", child),
      in("result", child),
      in("signature", att("AggStateSig").*),
    )

    r += node(
      Agg,
      "CombOp",
      in("i1", att("Int")),
      in("i2", att("Int")),
      in("aggSig", att("PhysicalAggSig")),
    )

    r += node(Agg, "ResultOp", in("idx", att("Int")), in("aggSig", att("PhysicalAggSig")))
      .withCompanionExtension

    r += node(Agg, "AggStateValue", in("i", att("Int")), in("aggSig", att("AggStateSig")))

    r += node(
      Agg,
      "SerializeAggs",
      in("startIdx", att("Int")),
      in("serializedIdx", att("Int")),
      in("spec", att("BufferSpec")),
      in("aggSigs", att("AggStateSig").*),
    )

    r += node(
      Agg,
      "DeserializeAggs",
      in("startIdx", att("Int")),
      in("serializedIdx", att("Int")),
      in("spec", att("BufferSpec")),
      in("aggSigs", att("AggStateSig").*),
    )

    r += node(
      IO,
      "ReadPartition",
      in("context", child),
      in("rowType", att("TStruct")),
      in("reader", att("PartitionReader")),
    )

    r += node(
      IO,
      "WritePartition",
      in("value", child),
      in("writeCtx", child),
      in("writer", att("PartitionWriter")),
    )

    r += node(
      IO,
      "WriteMetadata",
      in("writeAnnotations", child),
      in("writer", att("MetadataWriter")),
    )

    r += node(
      IO,
      "ReadValue",
      in("path", child),
      in("reader", att("ValueReader")),
      in("requestedType", att("Type")),
    )

    r += node(
      IO,
      "WriteValue",
      in("value", child),
      in("path", child),
      in("writer", att("ValueWriter")),
    )

    r += node(
      NDArray,
      "MakeNDArray",
      in("data", child),
      in("shape", child),
      in("rowMajor", child),
      errorID,
    )
      .typed("TNDArray").withCompanionExtension

    r += node(NDArray, "NDArrayShape", in("nd", child))

    r += node(NDArray, "NDArrayReshape", in("nd", child), in("shape", child), errorID).typed(
      "TNDArray"
    )

    r += node(NDArray, "NDArrayConcat", in("nds", child), in("axis", att("Int"))).typed("TNDArray")
    r += node(NDArray, "NDArrayRef", in("nd", child), in("idxs", child.*), errorID)
    r += node(NDArray, "NDArraySlice", in("nd", child), in("slices", child)).typed("TNDArray")
    r += node(NDArray, "NDArrayFilter", in("nd", child), in("keep", child.*)).typed("TNDArray")

    r += node(NDArray, "NDArrayReindex", in("nd", child), in("indexExpr", att("Int").*))
      .typed("TNDArray")

    r += node(NDArray, "NDArrayAgg", in("nd", child), in("axes", att("Int").*)).typed("TNDArray")
    r += node(NDArray, "NDArrayWrite", in("nd", child), in("path", child)).typed("TVoid.type")
    r += node(NDArray, "NDArrayMatMul", in("l", child), in("r", child), errorID).typed("TNDArray")

    r += node(
      NDArray,
      "NDArrayQR",
      in("nd", child),
      in("mode", att("String")),
      errorID,
    ).withCompanionExtension

    r += node(
      NDArray,
      "NDArraySVD",
      in("nd", child),
      in("fullMatrices", att("Boolean")),
      in("computeUV", att("Boolean")),
      errorID,
    )
      .withCompanionExtension

    r += node(NDArray, "NDArrayEigh", in("nd", child), in("eigvalsOnly", att("Boolean")), errorID)
      .withCompanionExtension

    r += node(NDArray, "NDArrayInv", in("nd", child), errorID).typed(
      "TNDArray"
    ).withCompanionExtension

    r += node(NDArray, "NDArrayMap", in("nd", child), in("valueName", name), in("body", child))
      .typed("TNDArray")

    r += node(
      NDArray,
      "NDArrayMap2",
      in("l", child),
      in("r", child),
      in("lName", name),
      in("rName", name),
      in("body", child),
      errorID,
    )
      .typed("TNDArray")

    r += node(Random, "RNGSplit", in("state", child), in("dynBitstring", child))
    r += node(Random, "RNGSplitStatic", in("state", child), in("staticUid", att("Long")))
    r += node(Random, "RNGStateLiteral")

    r += node(Relational, "RelationalRef", name, _typ()).withTraits(BaseRef, Atom)
    r += node(Relational, "RelationalLet", name, in("value", child), in("body", child))

    r += handWritten(Table, "TableParallelize")
    r += handWritten(Table, "TableKeyBy")
    r += handWritten(Table, "TableJoin")
    r += handWritten(Table, "TableMultiWayZipJoin")
    r += handWritten(Table, "TableLeftJoinRightDistinct")
    r += handWritten(Table, "TableExplode")
    r += handWritten(Table, "TableUnion")
    r += handWritten(Table, "TableRename")
    r += node(Table, "TableAggregate", tableChild, in("query", child))

    r += node(
      Table,
      "TableMultiWrite",
      in("_children", tableChild.*),
      in("writer", att("WrappedMatrixNativeMultiWriter")),
    )

    r += node(Table, "TableCollect", tableChild)
    r += handWritten(Table, "TableGen")
    r += handWritten(Table, "TableMapPartitions")
    r += handWritten(Table, "TableMapRows")
    r += handWritten(Table, "TableKeyByAndAggregate")
    r += handWritten(Table, "TableAggregateByKey")
    r += node(Table, "TableCount", tableChild)
    r += node(Table, "TableWrite", tableChild, in("writer", att("TableWriter")))
    r += node(Table, "TableGetGlobals", tableChild)

    r += node(
      Table,
      "TableToValueApply",
      tableChild,
      in("function", att("TableToValueFunction")),
    )

    r += handWritten(Table, "TableFilter")
    r += handWritten(Table, "TableMapGlobals")
    r += handWritten(Table, "RelationalLetTable")
    r += handWritten(Table, "TableLiteral")
    r += handWritten(Table, "TableRead")
    r += handWritten(Table, "TableRange")
    r += handWritten(Table, "TableHead")
    r += handWritten(Table, "TableTail")
    r += handWritten(Table, "TableRepartition")
    r += handWritten(Table, "TableIntervalJoin")
    r += handWritten(Table, "TableDistinct")
    r += handWritten(Table, "TableOrderBy")
    r += handWritten(Table, "TableFilterIntervals")
    r += handWritten(Table, "TableToTableApply")
    r += handWritten(Table, "BlockMatrixToTable")
    r += handWritten(Table, "BlockMatrixToTableApply")

    r += handWritten(Matrix, "MatrixUnionCols")
    r += handWritten(Matrix, "MatrixKeyRowsBy")
    r += handWritten(Matrix, "MatrixAnnotateColsTable")
    r += handWritten(Matrix, "MatrixAnnotateRowsTable")
    r += handWritten(Matrix, "MatrixUnionRows")
    r += handWritten(Matrix, "CastTableToMatrix")
    r += handWritten(Matrix, "MatrixRename")
    r += node(Matrix, "MatrixAggregate", matrixChild, in("query", child))

    r += node(
      Matrix,
      "MatrixMultiWrite",
      in("_children", matrixChild.*),
      in("writer", att("MatrixNativeMultiWriter")),
    )

    r += handWritten(Matrix, "MatrixAggregateRowsByKey")
    r += handWritten(Matrix, "MatrixAggregateColsByKey")
    r += node(Matrix, "MatrixCount", matrixChild)
    r += node(Matrix, "MatrixWrite", matrixChild, in("writer", att("MatrixWriter")))

    r += node(
      Matrix,
      "MatrixToValueApply",
      matrixChild,
      in("function", att("MatrixToValueFunction")),
    )

    r += handWritten(Matrix, "MatrixFilterCols")
    r += handWritten(Matrix, "MatrixFilterRows")
    r += handWritten(Matrix, "MatrixMapEntries")
    r += handWritten(Matrix, "MatrixMapRows")
    r += handWritten(Matrix, "MatrixMapCols")
    r += handWritten(Matrix, "MatrixMapGlobals")
    r += handWritten(Matrix, "MatrixFilterEntries")
    r += handWritten(Matrix, "RelationalLetMatrixTable")
    r += handWritten(Matrix, "MatrixRowsTable")
    r += handWritten(Matrix, "MatrixColsTable")
    r += handWritten(Matrix, "MatrixEntriesTable")
    r += handWritten(Matrix, "CastMatrixToTable")
    r += handWritten(Matrix, "MatrixToTableApply")
    r += handWritten(Matrix, "MatrixLiteral")
    r += handWritten(Matrix, "MatrixRead")
    r += handWritten(Matrix, "MatrixChooseCols")
    r += handWritten(Matrix, "MatrixCollectColsByKey")
    r += handWritten(Matrix, "MatrixExplodeRows")
    r += handWritten(Matrix, "MatrixRepartition")
    r += handWritten(Matrix, "MatrixDistinctByRow")
    r += handWritten(Matrix, "MatrixRowsHead")
    r += handWritten(Matrix, "MatrixColsHead")
    r += handWritten(Matrix, "MatrixRowsTail")
    r += handWritten(Matrix, "MatrixColsTail")
    r += handWritten(Matrix, "MatrixExplodeCols")
    r += handWritten(Matrix, "MatrixToMatrixApply")
    r += handWritten(Matrix, "MatrixFilterIntervals")

    r += handWritten(BlockMatrix, "BlockMatrixBroadcast")
    r += handWritten(BlockMatrix, "ValueToBlockMatrix")
    r += handWritten(BlockMatrix, "BlockMatrixMap")
    r += handWritten(BlockMatrix, "BlockMatrixMap2")

    r += node(
      BlockMatrix,
      "BlockMatrixToValueApply",
      bmChild,
      in("function", att("BlockMatrixToValueFunction")),
    )

    r += node(BlockMatrix, "BlockMatrixCollect", bmChild)
    r += node(BlockMatrix, "BlockMatrixWrite", bmChild, in("writer", att("BlockMatrixWriter")))

    r += node(
      BlockMatrix,
      "BlockMatrixMultiWrite",
      in("blockMatrices", bmChild.*),
      in("writer", att("BlockMatrixMultiWriter")),
    )

    r += handWritten(BlockMatrix, "BlockMatrixRead")
    r += handWritten(BlockMatrix, "BlockMatrixDot")
    r += handWritten(BlockMatrix, "BlockMatrixAgg")
    r += handWritten(BlockMatrix, "BlockMatrixFilter")
    r += handWritten(BlockMatrix, "BlockMatrixDensify")
    r += handWritten(BlockMatrix, "BlockMatrixSparsify")
    r += handWritten(BlockMatrix, "BlockMatrixSlice")
    r += handWritten(BlockMatrix, "BlockMatrixRandom")

    r.result()
  }

  /* `Tag_gen.scala`: the `Family` and `Tag` types, and one constant per family and per node.
   * Constants are untyped `final val` Int literals so that `@switch` matches on them compile to
   * jump tables. */
  private def tagFile(nodes: Seq[(Family, String)]): String = {
    val byFamily = nodes.groupBy(_._1)
    // A tag is stored as a Short, so that BaseIR's tag field fits in object padding.
    require(
      (Family.values.size << FamilyShift) <= Short.MaxValue + 1,
      s"${Family.values.size} families do not fit in a Short tag",
    )
    val familyConsts = Family.values.toSeq.map(f => s"  final val $f = ${f.id}")
    val tagConsts = Family.values.toSeq.flatMap { f =>
      val members = byFamily.getOrElse(f, Seq.empty).map(_._2)
      require(
        members.length <= (1 << FamilyShift),
        s"family $f has more than ${1 << FamilyShift} nodes",
      )
      s"\n  // $f" +: members.zipWithIndex.map { case (n, j) =>
          s"  final val $n = ${(f.id << FamilyShift) | j}"
        }
    }
    s"""// Generated by ir-gen; see `Family` in ir-gen.
       |package is.hail.expr.ir
       |
       |// The group an IR node's class belongs to.
       |final class Family private[ir] (val toInt: Int) extends AnyVal
       |
       |object Family {
       |${familyConsts.mkString("\n")}
       |}
       |
       |// Identifies an IR node's class. The bits above `Tag.FamilyShift` hold its [[Family]].
       |// Stored as a Short so that `BaseIR.tag` fits in the object padding most node classes
       |// already have.
       |final class Tag private[ir] (private val repr: Short) extends AnyVal {
       |  def toInt: Int = repr.toInt
       |  def family: Family = new Family(toInt >>> Tag.FamilyShift)
       |}
       |
       |object Tag {
       |  final val FamilyShift = $FamilyShift
       |${tagConsts.mkString("\n")}
       |}
       |""".stripMargin
  }

  @main
  def main(path: String): Unit = {
    val pack = "package is.hail.expr.ir.defs"
    val imports = Seq(
      "is.hail.annotations.Annotation",
      "is.hail.collection.FastSeq",
      "is.hail.io.{AbstractTypedCodecSpec, BufferSpec}",
      "is.hail.types.virtual.{Type, TArray, TNDArray, TStream, TVoid, TStruct, TTuple}",
      "is.hail.utils.StringEscapeUtils",
      "is.hail.expr.ir.{AggOp, BaseIR, IR, TableIR, MatrixIR, BlockMatrixIR, Name, UnaryOp, BinaryOp, " +
        "ComparisonOp, CanEmit, EmitParamType, TableWriter, " +
        "WrappedMatrixNativeMultiWriter, MatrixWriter, MatrixNativeMultiWriter, BlockMatrixWriter, " +
        "BlockMatrixMultiWriter, ValueReader, ValueWriter, Tag}",
      "is.hail.expr.ir.lowering.TableStageDependency",
      "is.hail.expr.ir.agg.{PhysicalAggSig, AggStateSig}",
      "is.hail.expr.ir.functions.{UnseededMissingnessAwareJVMFunction, " +
        "UnseededMissingnessObliviousJVMFunction, TableToValueFunction, MatrixToValueFunction, " +
        "BlockMatrixToValueFunction}",
      "is.hail.expr.ir.defs.exts._",
    )
    val nodes = allNodes
    val gen = pack + "\n\n" + imports.map(i => s"import $i").mkString("\n") + "\n\n" +
      nodes.filterNot(_.isHandWritten).map(_.generateDef).mkString("\n")
    os.write(os.Path(path) / "IR_gen.scala", gen)
    val tagged = nodes.map(n => (n.family, n.name))
    os.write(os.Path(path) / "Tag_gen.scala", tagFile(tagged))
  }

  def main(args: Array[String]): Unit = {
    val _ = ParserForMethods(this).runOrExit(ArraySeq.unsafeWrapArray(args))
  }
}
