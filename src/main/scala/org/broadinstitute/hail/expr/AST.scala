package org.broadinstitute.hail.expr

import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.variant.{Sample, Variant, Genotype}

object Type {
  val sampleFields = Map(
    "id" -> TString)

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

case object TCharacter extends Type

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

case object TSample extends TAbstractStruct(Type.sampleFields)

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

  def evalFlatCompose[T](c: EvalContext, subexpr: AST)
    (g: (T) => Any): () => Any = {
    val f = subexpr.eval(c)
    () => {
      val x = f()
      if (x != null)
        g(x.asInstanceOf[T])
      else
        null
    }
  }

  def evalCompose[T](c: EvalContext, subexpr: AST)
    (g: (T) => Any): () => Any = {
    val f = subexpr.eval(c)
    () => {
      val x = f()
      if (x != null)
        g(x.asInstanceOf[T])
      else
        null
    }
  }

  def evalCompose[T1, T2](c: EvalContext, subexpr1: AST, subexpr2: AST)
    (g: (T1, T2) => Any): () => Any = {
    val f1 = subexpr1.eval(c)
    val f2 = subexpr2.eval(c)
    () => {
      val x = f1()
      if (x != null) {
        val y = f2()
        if (y != null)
          g(x.asInstanceOf[T1], y.asInstanceOf[T2])
        else
          null
      } else
        null
    }
  }

  def evalComposeNumeric[T](c: EvalContext, subexpr: AST)
    (g: (T) => Any)
    (implicit convT: NumericConversion[T]): () => Any = {
    val f = subexpr.eval(c)
    () => {
      val x = f()
      if (x != null)
        g(x.asInstanceOf[T])
      else
        null
    }
  }

  def evalComposeNumeric[T1, T2](c: EvalContext, subexpr1: AST, subexpr2: AST)
    (g: (T1, T2) => Any)
    (implicit convT1: NumericConversion[T1], convT2: NumericConversion[T2]): () => Any = {
    val f1 = subexpr1.eval(c)
    val f2 = subexpr2.eval(c)
    () => {
      val x = f1()
      if (x != null) {
        val y = f2()
        if (y != null)
          g(convT1.to(x), convT2.to(y))
        else
          null
      } else
        null
    }
  }
}

// FIXME LexPos
sealed abstract class AST(subexprs: Array[AST] = Array.empty) extends Serializable {
  var `type`: Type = null

  def this(subexpr1: AST) = this(Array(subexpr1))

  def this(subexpr1: AST, subexpr2: AST) = this(Array(subexpr1, subexpr2))

  def eval(c: EvalContext): () => Any

  def typecheckThis(typeSymTab: SymbolTable): Type = typecheckThis()

  def typecheckThis(): Type = throw new UnsupportedOperationException

  def typecheck(typeSymTab: SymbolTable) {
    subexprs.foreach(_.typecheck(typeSymTab))
    `type` = typecheckThis(typeSymTab)
  }
}

case class Const(value: Any, t: Type) extends AST {
  def eval(c: EvalContext): () => Any = {
    val v = value
    () => v
  }

  override def typecheckThis(): Type = t
}

case class Select(lhs: AST, rhs: String) extends AST(lhs) {
  override def typecheckThis(): Type = {
    (lhs.`type`, rhs) match {
      case (TSample, "id") => TString
      case (TGenotype, "gt") => TInt
      case (TGenotype, "ad") => TArray(TInt)
      case (TGenotype, "dp") => TInt
      case (TGenotype, "gq") => TInt
      case (TGenotype, "pl") => TArray(TInt)
      case (TVariant, "contig") => TString
      case (TVariant, "start") => TInt
      case (TVariant, "ref") => TString
      case (TVariant, "alt") => TString
      case (TStruct(fields), _) => fields(rhs)
      case (t: TNumeric, "toInt") => TInt
      case (t: TNumeric, "toLong") => TLong
      case (t: TNumeric, "toFloat") => TFloat
      case (t: TNumeric, "toDouble") => TDouble
    }
  }

  def eval(c: EvalContext): () => Any = (lhs.`type`, rhs) match {
    case (TSample, "id") =>
      AST.evalCompose[Sample](c, lhs)(_.id)

    case (TGenotype, "gt") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.call.map(_.gt))
    case (TGenotype, "ad") =>
      AST.evalCompose[Genotype](c, lhs)(_.ad)
    case (TGenotype, "dp") =>
      AST.evalCompose[Genotype](c, lhs)(_.dp)
    case (TGenotype, "gq") =>
      AST.evalCompose[Genotype](c, lhs)(_.gq)
    case (TGenotype, "pl") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.call.map(_.pl))

    case (TVariant, "contig") =>
      AST.evalCompose[Variant](c, lhs)(_.contig)
    case (TVariant, "start") =>
      AST.evalCompose[Variant](c, lhs)(_.start)
    case (TVariant, "ref") =>
      AST.evalCompose[Variant](c, lhs)(_.ref)
    case (TVariant, "alt") =>
      AST.evalCompose[Variant](c, lhs)(_.alt)

    case (TStruct(fields), _) =>
      AST.evalFlatCompose[Map[String, Any]](c, lhs) { m =>
        val x = m.getOrElse(rhs, null)
        if (x != null
          && x.isInstanceOf[Annotations])
          x.asInstanceOf[Annotations].attrs
        else
          x
      }

    case (TInt, "toInt") => lhs.eval(c)
    case (TInt, "toLong") => AST.evalCompose[Int](c, lhs)(_.toLong)
    case (TInt, "toFloat") => AST.evalCompose[Int](c, lhs)(_.toFloat)
    case (TInt, "toDouble") => AST.evalCompose[Int](c, lhs)(_.toDouble)

    case (TLong, "toInt") => AST.evalCompose[Long](c, lhs)(_.toInt)
    case (TLong, "toLong") => lhs.eval(c)
    case (TLong, "toFloat") => AST.evalCompose[Long](c, lhs)(_.toFloat)
    case (TLong, "toDouble") => AST.evalCompose[Long](c, lhs)(_.toDouble)

    case (TFloat, "toInt") => AST.evalCompose[Float](c, lhs)(_.toInt)
    case (TFloat, "toLong") => AST.evalCompose[Float](c, lhs)(_.toLong)
    case (TFloat, "toFloat") => lhs.eval(c)
    case (TFloat, "toDouble") => AST.evalCompose[Float](c, lhs)(_.toDouble)

    case (TDouble, "toInt") => AST.evalCompose[Double](c, lhs)(_.toInt)
    case (TDouble, "toLong") => AST.evalCompose[Double](c, lhs)(_.toLong)
    case (TDouble, "toFloat") => AST.evalCompose[Double](c, lhs)(_.toFloat)
    case (TDouble, "toDouble") => lhs.eval(c)

    case (TString, "toInt") => AST.evalCompose[String](c, lhs)(_.toInt)
    case (TString, "toLong") => AST.evalCompose[String](c, lhs)(_.toLong)
    case (TString, "toFloat") => AST.evalCompose[String](c, lhs)(_.toFloat)
    case (TString, "toDouble") => AST.evalCompose[String](c, lhs)(_.toDouble)
  }
}

case class BinaryOp(lhs: AST, operation: String, rhs: AST) extends AST(lhs, rhs) {
  def eval(c: EvalContext): () => Any = (operation, `type`) match {
    case ("+", TString) => AST.evalCompose[String, String](c, lhs, rhs)(_ + _)

    case ("||", TBoolean) => AST.evalCompose[Boolean, Boolean](c, lhs, rhs)(_ || _)
    case ("&&", TBoolean) => AST.evalCompose[Boolean, Boolean](c, lhs, rhs)(_ && _)

    case ("+", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ + _)
    case ("-", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ - _)
    case ("*", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ * _)
    case ("/", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ / _)
    case ("%", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ % _)

    case ("+", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ + _)
    case ("-", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ - _)
    case ("*", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ * _)
    case ("/", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ / _)
  }

  override def typecheckThis(): Type = (lhs.`type`, operation, rhs.`type`) match {
    case (TString, "+", TString) => TString
    case (TBoolean, "||", TBoolean) => TBoolean
    case (TBoolean, "&&", TBoolean) => TBoolean
    case (lhsType: TNumeric, "+", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
  }
}

case class Comparison(lhs: AST, operation: String, rhs: AST) extends AST(lhs, rhs) {
  var operandType: Type = null

  def eval(c: EvalContext): () => Any = (operation, operandType) match {
    // FIXME common sueprtype for Array, Set
    case ("==", TBoolean) => AST.evalCompose[Boolean, Boolean](c, lhs, rhs)(_ == _)
    case ("!=", TBoolean) => AST.evalCompose[Boolean, Boolean](c, lhs, rhs)(_ != _)

    case ("==", TInt) => AST.evalCompose[Int, Int](c, lhs, rhs)(_ == _)
    case ("!=", TInt) => AST.evalCompose[Int, Int](c, lhs, rhs)(_ != _)
    case ("<", TInt) => AST.evalCompose[Int, Int](c, lhs, rhs)(_ < _)
    case ("<=", TInt) => AST.evalCompose[Int, Int](c, lhs, rhs)(_ <= _)
    case (">", TInt) => AST.evalCompose[Int, Int](c, lhs, rhs)(_ > _)
    case (">=", TInt) => AST.evalCompose[Int, Int](c, lhs, rhs)(_ >= _)

    case ("==", TDouble) => AST.evalCompose[Double, Double](c, lhs, rhs)(_ == _)
    case ("!=", TDouble) => AST.evalCompose[Double, Double](c, lhs, rhs)(_ != _)
    case ("<", TDouble) => AST.evalCompose[Double, Double](c, lhs, rhs)(_ < _)
    case ("<=", TDouble) => AST.evalCompose[Double, Double](c, lhs, rhs)(_ <= _)
    case (">", TDouble) => AST.evalCompose[Double, Double](c, lhs, rhs)(_ > _)
    case (">=", TDouble) => AST.evalCompose[Double, Double](c, lhs, rhs)(_ >= _)
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
  def eval(c: EvalContext): () => Any = (operation, `type`) match {
    case ("-", TInt) => AST.evalComposeNumeric[Int](c, operand)(-_)
    case ("-", TLong) => AST.evalComposeNumeric[Long](c, operand)(-_)
    case ("-", TFloat) => AST.evalComposeNumeric[Float](c, operand)(-_)
    case ("-", TDouble) => AST.evalComposeNumeric[Double](c, operand)(-_)

    case ("!", TBoolean) => AST.evalCompose[Boolean](c, operand)(!_)
  }

  override def typecheckThis(): Type = (operation, operand.`type`) match {
    case ("-", t: TNumeric) => AST.promoteNumeric(t)
    case ("!", TBoolean) => TBoolean
  }
}

case class Invoke(function: String, args: Array[AST])

case class SymRef(symbol: String) extends AST {
  def eval(c: EvalContext): () => Any = {
    val i = c._1(symbol)._1
    val a = c._2
    () => a(i)
  }

  override def typecheckThis(typeSymTab: SymbolTable): Type =
    typeSymTab(symbol)._2
}
