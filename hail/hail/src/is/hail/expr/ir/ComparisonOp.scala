package is.hail.expr.ir

import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.orderings.CodeOrdering.F
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.virtual.Type

object ComparisonOp {

  def checkCompatible[T](lt: Type, rt: Type): Unit =
    if (lt != rt)
      throw new RuntimeException(s"Cannot compare types $lt and $rt")

  val fromString: PartialFunction[String, ComparisonOp[_]] = {
    case "==" | "EQ" => EQ
    case "!=" | "NEQ" => NEQ
    case ">=" | "GTEQ" => GTEQ
    case "<=" | "LTEQ" => LTEQ
    case ">" | "GT" => GT
    case "<" | "LT" => LT
    case "Compare" => Compare
  }

  def negate(op: ComparisonOp[Boolean]): ComparisonOp[Boolean] = {
    op match {
      case GT => LTEQ
      case LT => GTEQ
      case GTEQ => LT
      case LTEQ => GT
      case EQ => NEQ
      case NEQ => EQ
      case EQWithNA => NEQWithNA
      case NEQWithNA => EQWithNA
      case StructLT(sortFields) => StructGTEQ(sortFields)
      case StructGT(sortFields) => StructLTEQ(sortFields)
      case StructLTEQ(sortFields) => StructGT(sortFields)
      case StructGTEQ(sortFields) => StructLT(sortFields)
    }
  }

  def swap(op: ComparisonOp[Boolean]): ComparisonOp[Boolean] = {
    op match {
      case GT => LT
      case LT => GT
      case GTEQ => LTEQ
      case LTEQ => GTEQ
      case EQ => EQ
      case NEQ => NEQ
      case EQWithNA => EQWithNA
      case NEQWithNA => NEQWithNA
      case StructLT(sortFields) => StructGT(sortFields)
      case StructGT(sortFields) => StructLT(sortFields)
      case StructLTEQ(sortFields) => StructGTEQ(sortFields)
      case StructGTEQ(sortFields) => StructLTEQ(sortFields)
    }
  }
}

sealed trait ComparisonOp[ReturnType] {
  val op: CodeOrdering.Op
  val strict: Boolean = true

  def codeOrdering(ecb: EmitClassBuilder[_], t1p: SType, t2p: SType)
    : CodeOrdering.F[op.ReturnType] = {
    ComparisonOp.checkCompatible(t1p.virtualType, t2p.virtualType)
    ecb.getOrderingFunction(t1p, t2p, op).asInstanceOf[CodeOrdering.F[op.ReturnType]]
  }

  def render(): is.hail.utils.prettyPrint.Doc = Pretty.prettyClass(this)
}

case object GT extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Gt()
}

case object GTEQ extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Gteq()
}

case object LTEQ extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Lteq()
}

case object LT extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Lt()
}

case object EQ extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Equiv()
}

case object NEQ extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Neq()
}

case object EQWithNA extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Equiv()
  override val strict: Boolean = false
}

case object NEQWithNA extends ComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.Neq()
  override val strict: Boolean = false
}

case object Compare extends ComparisonOp[Int] {
  override val strict: Boolean = false
  val op: CodeOrdering.Op = CodeOrdering.Compare()
}

trait StructComparisonOp[T] extends ComparisonOp[T] {
  val sortFields: IndexedSeq[SortField]

  override def codeOrdering(ecb: EmitClassBuilder[_], t1: SType, t2: SType): F[op.ReturnType] = {
    ComparisonOp.checkCompatible(t1.virtualType, t2.virtualType)
    ecb.getStructOrderingFunction(
      t1.asInstanceOf[SBaseStruct],
      t2.asInstanceOf[SBaseStruct],
      sortFields,
      op,
    ).asInstanceOf[CodeOrdering.F[op.ReturnType]]
  }
}

case class StructCompare(sortFields: IndexedSeq[SortField]) extends StructComparisonOp[Int] {
  val op: CodeOrdering.Op = CodeOrdering.StructCompare()
  override val strict: Boolean = false
}

case class StructLT(sortFields: IndexedSeq[SortField]) extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructLt()
}

case class StructLTEQ(sortFields: IndexedSeq[SortField]) extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructLteq()
}

case class StructGT(sortFields: IndexedSeq[SortField]) extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructGt()
}

case class StructGTEQ(sortFields: IndexedSeq[SortField]) extends StructComparisonOp[Boolean] {
  val op: CodeOrdering.Op = CodeOrdering.StructGteq()
}
