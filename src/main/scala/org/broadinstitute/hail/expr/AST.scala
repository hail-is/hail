package org.broadinstitute.hail.expr

import org.broadinstitute.hail.variant.{Variant, Genotype}

object Type {
  val genotypeFields = Map(
    "gt" -> TInt,
    "ad" -> TArray(TInt),
    "dp" -> TInt,
    "gq" -> TInt,
    "pl" -> TArray(TInt))

  val variantFields = Map(
    "chrom" -> TString,
    "start" -> TInt,
    "ref" -> TString,
    "alt" -> TString)
}

trait NumericConversion[T] {
  def to(numeric: Any): T
}

object IntNumericConversion extends NumericConversion[Int] {
  def to(numeric: Any): Int = numeric match {
    case i: Int => i
  }
}

object LongNumericConversion extends NumericConversion[Long] {
  def to(numeric: Any): Long = numeric match {
    case i: Int => i
    case l: Long => l
  }
}

object FloatNumericConversion extends NumericConversion[Float] {
  def to(numeric: Any): Float = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
  }
}

object DoubleNumericConversion extends NumericConversion[Double] {
  def to(numeric: Any): Double = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
    case d: Double => d
  }
}

sealed abstract class Type extends Serializable

case object TBoolean extends Type

abstract class TNumeric extends Type

case object TInt extends TNumeric

case object TLong extends TNumeric

case object TFloat extends TNumeric

case object TDouble extends TNumeric

case object TUnit extends Type

case object TString extends Type

case class TArray(elementType: Type) extends Type

case class TSet(elementType: Type) extends Type

// FIXME name?
class TAbstractStruct(fields: Map[String, Type]) extends Type

case object TGenotype extends TAbstractStruct(Type.genotypeFields)

case object TVariant extends TAbstractStruct(Type.variantFields)

case class TStruct(fields: Map[String, Type]) extends TAbstractStruct(fields)

object AST extends Serializable {
  def promoteNumeric(t: TNumeric): Type = t

  def promoteNumeric(lhs: TNumeric, rhs: TNumeric): Type =
    if (lhs == TDouble || rhs == TDouble)
      TDouble
    else if (lhs == TFloat || rhs == TFloat)
      TFloat
    else if (lhs == TLong || rhs == TLong)
      TLong
    else
      TInt

  def evalFlatCompose[T](subexpr: AST)
    (g: (T) => Option[Any]): (SymbolTable) => Option[Any] = {
    symTab =>
      subexpr.eval(symTab).flatMap(x => g(x.asInstanceOf[T]))
  }

  def evalCompose[T](subexpr: AST)
    (g: (T) => Any): (SymbolTable) => Option[Any] = {
    symTab =>
      subexpr.eval(symTab).map(x => g(x.asInstanceOf[T]))
  }

  def evalCompose[T1, T2](subexpr1: AST, subexpr2: AST)
    (g: (T1, T2) => Any): (SymbolTable) => Option[Any] = {
    symTab =>
      subexpr1.eval(symTab).flatMap {
        x =>
          subexpr2.eval(symTab).map {
            y =>
              g(x.asInstanceOf[T1], y.asInstanceOf[T2])
          }
      }
  }

  def evalComposeNumeric[T](subexpr: AST)
    (g: (T) => Any)
    (implicit convT: NumericConversion[T]): (SymbolTable) => Option[Any] = {
    symTab =>
      subexpr.eval(symTab).map(x => g(convT.to(x)))
  }

  def evalComposeNumeric[T1, T2](subexpr1: AST, subexpr2: AST)
    (g: (T1, T2) => Any)
    (implicit convT1: NumericConversion[T1], convT2: NumericConversion[T2]): (SymbolTable) => Option[Any] = {
    symTab =>
      subexpr1.eval(symTab).flatMap {
        x =>
          subexpr2.eval(symTab).map {
            y =>
              g(convT1.to(x), convT2.to(y))
          }
      }
  }
}

// FIXME LexPos
sealed abstract class AST(subexprs: Array[AST] = Array.empty) extends Serializable {
  type SymbolTable = Map[String, Option[Any]]

  var `type`: Type = null

  def this(subexpr1: AST) = this(Array(subexpr1))

  def this(subexpr1: AST, subexpr2: AST) = this(Array(subexpr1, subexpr2))

  def eval: SymbolTable => Option[Any]

  def typecheckThis(typeSymTab: TypeSymbolTable): Type = typecheckThis()

  def typecheckThis(): Type = throw new UnsupportedOperationException

  def typecheck(typeSymTab: TypeSymbolTable) {
    subexprs.foreach(_.typecheck(typeSymTab))
    `type` = typecheckThis(typeSymTab)
  }
}

case class Const(value: Any, t: Type) extends AST {
  def eval = _ => Some(value)

  override def typecheckThis(): Type = t
}

case class Select(lhs: AST, rhs: String) extends AST(lhs) {
  override def typecheckThis(): Type = {
    (lhs.`type`, rhs) match {
      case (TGenotype, "gt") => TInt
      case (TStruct(fields), _) => fields(rhs)
      case (t: TNumeric, "toInt") => TInt
      case (t: TNumeric, "toLong") => TLong
      case (t: TNumeric, "toFloat") => TFloat
      case (t: TNumeric, "toDouble") => TDouble
    }
  }

  def eval = (lhs.`type`, rhs) match {
    case (TGenotype, "gt") =>
      AST.evalFlatCompose[Genotype](lhs)(_.call.map(_.gt))

    case (TStruct(fields), _) =>
      AST.evalFlatCompose[Map[String, Any]](lhs)(_.get(rhs))

    case (TInt, "toInt") => AST.evalCompose[Int](lhs)(identity)
    case (TInt, "toLong") => AST.evalCompose[Int](lhs)(_.toLong)
    case (TInt, "toFloat") => AST.evalCompose[Int](lhs)(_.toFloat)
    case (TInt, "toDouble") => AST.evalCompose[Int](lhs)(_.toDouble)

    case (TLong, "toInt") => AST.evalCompose[Long](lhs)(_.toInt)
    case (TLong, "toLong") => lhs.eval
    case (TLong, "toFloat") => AST.evalCompose[Long](lhs)(_.toFloat)
    case (TLong, "toDouble") => AST.evalCompose[Long](lhs)(_.toDouble)

    case (TFloat, "toInt") => AST.evalCompose[Float](lhs)(_.toInt)
    case (TFloat, "toLong") => AST.evalCompose[Float](lhs)(_.toLong)
    case (TFloat, "toFloat") => lhs.eval
    case (TFloat, "toDouble") => AST.evalCompose[Float](lhs)(_.toDouble)

    case (TDouble, "toInt") => AST.evalCompose[Double](lhs)(_.toInt)
    case (TDouble, "toLong") => AST.evalCompose[Double](lhs)(_.toLong)
    case (TDouble, "toFloat") => AST.evalCompose[Double](lhs)(_.toFloat)
    case (TDouble, "toDouble") => lhs.eval

    case (TString, "toInt") => AST.evalCompose[String](lhs)(_.toInt)
    case (TString, "toLong") => AST.evalCompose[String](lhs)(_.toLong)
    case (TString, "toFloat") => AST.evalCompose[String](lhs)(_.toFloat)
    case (TString, "toDouble") => AST.evalCompose[String](lhs)(_.toDouble)
  }
}

case class BinaryOp(lhs: AST, operation: String, rhs: AST) extends AST(lhs, rhs) {
  def eval = (operation, `type`) match {
    case ("+", TString) => AST.evalCompose[String, String](lhs, rhs)(_ + _)

    case ("+", TInt) => AST.evalComposeNumeric[Int, Int](lhs, rhs)(_ + _)
    case ("-", TInt) => AST.evalComposeNumeric[Int, Int](lhs, rhs)(_ - _)
    case ("*", TInt) => AST.evalComposeNumeric[Int, Int](lhs, rhs)(_ * _)
    case ("/", TInt) => AST.evalComposeNumeric[Int, Int](lhs, rhs)(_ / _)
    case ("%", TInt) => AST.evalComposeNumeric[Int, Int](lhs, rhs)(_ % _)

    case ("+", TDouble) => AST.evalComposeNumeric[Double, Double](lhs, rhs)(_ + _)
    case ("-", TDouble) => AST.evalComposeNumeric[Double, Double](lhs, rhs)(_ - _)
    case ("*", TDouble) => AST.evalComposeNumeric[Double, Double](lhs, rhs)(_ * _)
    case ("/", TDouble) => AST.evalComposeNumeric[Double, Double](lhs, rhs)(_ / _)
  }

  override def typecheckThis(): Type = (lhs.`type`, operation, rhs.`type`) match {
    case (TString, "+", TString) => TString
    case (lhsType: TNumeric, "+", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
  }
}

case class Comparison(lhs: AST, operation: String, rhs: AST) extends AST(lhs, rhs) {
  var operandType: Type = null

  def eval = (operation, operandType) match {
      // FIXME common sueprtype for Array, Set
    case ("==", TBoolean) => AST.evalCompose[Boolean, Boolean](lhs, rhs)(_ == _)
    case ("!=", TBoolean) => AST.evalCompose[Boolean, Boolean](lhs, rhs)(_ != _)

    case ("==", TInt) => AST.evalCompose[Int, Int](lhs, rhs)(_ == _)
    case ("!=", TInt) => AST.evalCompose[Int, Int](lhs, rhs)(_ != _)
    case ("<", TInt) => AST.evalCompose[Int, Int](lhs, rhs)(_ < _)
    case ("<=", TInt) => AST.evalCompose[Int, Int](lhs, rhs)(_ <= _)
    case (">", TInt) => AST.evalCompose[Int, Int](lhs, rhs)(_ > _)
    case (">=", TInt) => AST.evalCompose[Int, Int](lhs, rhs)(_ >= _)
  }

  override def typecheckThis(): Type = {
    operandType = (lhs.`type`, operation, rhs.`type`) match {
      case (TBoolean, "==" | "!=", TBoolean) => TBoolean
      case (TString, _, TString) => TString
      case (lhsType: TNumeric, _, rhsType: TNumeric) =>
        AST.promoteNumeric(lhsType, rhsType)
    }

    TBoolean
  }
}

case class UnaryOp(operation: String, operand: AST) extends AST(operand) {
  def eval = (operation, `type`) match {
    case ("-", TInt) => AST.evalComposeNumeric[Int](operand)(-_)
    case ("-", TLong) => AST.evalComposeNumeric[Long](operand)(-_)
    case ("-", TFloat) => AST.evalComposeNumeric[Float](operand)(-_)
    case ("-", TDouble) => AST.evalComposeNumeric[Double](operand)(-_)

    case ("!", TBoolean) => AST.evalCompose[Boolean](operand)(!_)
  }

  override def typecheckThis(): Type = (operation, operand.`type`) match {
    case ("-", t: TNumeric) => AST.promoteNumeric(t)
    case ("!", TBoolean) => TBoolean
  }
}

case class Invoke(function: String, args: Array[AST])

case class SymRef(symbol: String) extends AST {
  def eval = symTab => symTab.get(symbol)

  override def typecheckThis(typeSymTab: TypeSymbolTable): Type =
    typeSymTab(symbol)
}
