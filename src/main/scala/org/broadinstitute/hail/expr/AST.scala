package org.broadinstitute.hail.expr

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.Utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.variant.{Sample, AltAllele, Variant, Genotype}
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.parsing.input.{Position, Positional}

case class EvalContext(symTab: SymbolTable,
  a: ArrayBuffer[Any])

trait NumericConversion[T] extends Serializable {
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

sealed abstract class Type extends Serializable {
  def getAsOption[T](fields: String*)(implicit ct: ClassTag[T]): Option[T] = {
    getOption(fields: _*)
      .flatMap { t =>
        if (ct.runtimeClass.isInstance(t))
          Some(t.asInstanceOf[T])
        else
          None
      }
  }

  def getOption(fields: String*): Option[Type] = getOption(fields.toList)

  def getOption(path: List[String]): Option[Type] = {
    if (path.isEmpty)
      Some(this)
    else
      None
  }

  def delete(fields: String*): (Type, Deleter) = delete(fields.toList)

  def delete(path: List[String]): (Type, Deleter) = {
    if (path.nonEmpty)
      throw new AnnotationPathException()
    else
      (TEmpty, a => null)
  }

  def insert(signature: Type, fields: String*): (Type, Inserter) = insert(signature, fields.toList)

  def insert(signature: Type, path: List[String]): (Type, Inserter) = {
    if (path.nonEmpty)
      TStruct.empty.insert(signature, path)
    else
      (signature, (a, toIns) => toIns.orNull)
  }

  def query(fields: String*): Querier = query(fields.toList)

  def query(path: List[String]): Querier = {
    if (path.nonEmpty)
      throw new AnnotationPathException()
    else
      a => Option(a)
  }

  def pretty(sb: StringBuilder, indent: Int, path: Vector[String], arrayDepth: Int) {
    sb.append(" " * indent)
    sb.append(path.last)
    sb.append(": ")
    sb.append("Array[" * arrayDepth)
    sb.append(toString)
    sb.append("]" * arrayDepth)
    sb += '\n'
  }

  def fieldOption(fields: String*): Option[Field] = fieldOption(fields.toList)

  def fieldOption(path: List[String]): Option[Field] =
    None

  def typeCheck(a: Any): Boolean

  def schema: DataType
}

case object TEmpty extends Type {
  override def toString = "Empty"

  def typeCheck(a: Any): Boolean = a == null

  def schema =
  // placeholder
    BooleanType
}

case object TBoolean extends Type {
  override def toString = "Boolean"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Boolean]

  def schema = BooleanType
}

case object TChar extends Type {
  override def toString = "Char"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Character]

  def schema = throw new UnsupportedOperationException
}

abstract class TNumeric extends Type

case object TInt extends TNumeric {
  override def toString = "Int"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Int]

  def schema = IntegerType
}

case object TLong extends TNumeric {
  override def toString = "Long"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Long]

  def schema = LongType
}

case object TFloat extends TNumeric {
  override def toString = "Float"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Float]

  def schema = FloatType
}

case object TDouble extends TNumeric {
  override def toString = "Double"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Double]

  def schema = DoubleType
}

case object TString extends Type {
  override def toString = "String"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[String]

  def schema = StringType
}

case class TArray(elementType: Type) extends Type {
  override def toString = s"Array[$elementType]"

  override def pretty(sb: StringBuilder, indent: Int, path: Vector[String], arrayDepth: Int) {
    elementType.pretty(sb, indent, path, arrayDepth + 1)
  }

  def typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck))

  def schema = ArrayType(elementType.schema)
}

case class TSet(elementType: Type) extends Type {
  override def toString = s"Set[$elementType]"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck)

  def schema = ArrayType(elementType.schema)
}

case object TSample extends Type {
  override def toString = "Sample"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Sample]

  def schema = StructType(Array(
    StructField("id", StringType, nullable = false)))
}

case object TGenotype extends Type {
  override def toString = "Genotype"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Genotype]

  def schema = Genotype.schema
}

case object TAltAllele extends Type {
  override def toString = "AltAllele"

  def typeCheck(a: Any): Boolean = a == null || a == null || a.isInstanceOf[AltAllele]

  def schema = AltAllele.schema
}

case object TVariant extends Type {
  override def toString = "Variant"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Variant]

  def schema = Variant.schema
}

object TStruct {
  def empty: TStruct = TStruct(Array.empty[Field])

  def apply(args: (String, Type)*): TStruct =
    TStruct(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => Field(n, t, i) }
      .toArray)
}

case class Field(name: String, `type`: Type,
  index: Int,
  attrs: Map[String, String] = Map.empty) {
  def attr(s: String): Option[String] = attrs.get(s)
}

case class TStruct(fields: IndexedSeq[Field]) extends Type {
  val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  def selfField(name: String): Option[Field] = fieldIdx.get(name).map(i => fields(i))

  def size: Int = fields.length

  override def getOption(path: List[String]): Option[Type] =
    if (path.isEmpty)
      Some(this)
    else
      selfField(path.head).map(_.`type`).flatMap(t => t.getOption(path.tail))

  override def fieldOption(path: List[String]): Option[Field] =
    if (path.isEmpty)
      None
    else {
      val f = selfField(path.head)
      if (path.length == 1)
        f
      else
        f.flatMap(_.`type`.fieldOption(path.tail))
    }

  override def query(p: List[String]): Querier = {
    if (p.isEmpty)
      a => Option(a)
    else {
      selfField(p.head) match {
        case Some(f) =>
          val q = f.`type`.query(p.tail)
          val localIndex = f.index
          a =>
            if (a == null)
              None
            else
              q(a.asInstanceOf[Row].get(localIndex))
        case None => throw new AnnotationPathException()
      }
    }
  }

  override def delete(p: List[String]): (Type, Deleter) = {
    if (p.isEmpty)
      (TEmpty, a => null)
    else {
      val key = p.head
      val f = selfField(key) match {
        case Some(f) => f
        case None => throw new AnnotationPathException(s"$key not found")
      }
      val index = f.index
      val (newFieldType, d) = f.`type`.delete(p.tail)
      val newType: Type =
        if (newFieldType == TEmpty)
          deleteKey(key, f.index)
        else
          updateKey(key, f.index, newFieldType)

      val localDeleteFromRow = newFieldType == TEmpty

      val deleter: Deleter = { a =>
        if (a == null)
          null
        else {
          val r = a.asInstanceOf[Row]

          if (localDeleteFromRow)
            r.delete(index)
          else
            r.update(index, d(r.get(index)))
        }
      }
      (newType, deleter)
    }
  }

  override def insert(signature: Type, p: List[String]): (Type, Inserter) = {
    if (p.isEmpty)
      (signature, (a, toIns) => toIns.orNull)
    else {
      val key = p.head
      val f = selfField(key)
      val keyIndex = f.map(_.index)
      val (newKeyType, keyF) = f
        .map(_.`type`)
        .getOrElse(TStruct.empty)
        .insert(signature, p.tail)

      val newSignature = keyIndex match {
        case Some(i) => updateKey(key, i, newKeyType)
        case None => appendKey(key, newKeyType)
      }

      val localSize = fields.size

      val inserter: Inserter = (a, toIns) => {
        val r = if (a == null)
          Row.fromSeq(Array.fill[Any](localSize)(null))
        else
          a.asInstanceOf[Row]
        keyIndex match {
          case Some(i) => r.update(i, keyF(r.get(i), toIns))
          case None => r.append(keyF(null, toIns))
        }
      }
      (newSignature, inserter)
    }
  }

  def updateKey(key: String, i: Int, sig: Type): Type = {
    assert(fieldIdx.contains(key))

    val newFields = Array.fill[Field](fields.length)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(i) = Field(key, sig, i)
    TStruct(newFields)
  }

  def deleteKey(key: String, index: Int): Type = {
    assert(fieldIdx.contains(key))
    if (fields.length == 1)
      TEmpty
    else {
      val newFields = Array.fill[Field](fields.length - 1)(null)
      for (i <- 0 until index)
        newFields(i) = fields(i)
      for (i <- index + 1 until fields.length)
        newFields(i - 1) = fields(i).copy(index = i - 1)
      TStruct(newFields)
    }
  }

  def appendKey(key: String, sig: Type): TStruct = {
    assert(!fieldIdx.contains(key))
    val newFields = Array.fill[Field](fields.length + 1)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(fields.length) = Field(key, sig, fields.length)
    TStruct(newFields)
  }

  override def toString = "Struct"

  override def pretty(sb: StringBuilder, indent: Int, path: Vector[String], arrayDepth: Int) {
    sb.append(" " * indent)
    sb.append(path.last)
    sb.append(": ")
    path.foreachBetween { f => sb.append(f) } { () => sb += '.' }
    for (i <- 0 until arrayDepth)
      sb.append("[<index>]")
    sb.append(".<identifier>\n")
    for (f <- fields) {
      f.`type`.pretty(sb, indent + 2, path :+ f.name, 0)
      /*
      if (f.attrs.nonEmpty) {
        sb.append(" " * (indent + 2))
        f.attrs.foreachBetween { case (k, v) =>
          sb.append(k)
          sb += '='
          sb.append(v)
        } { () => sb.append(", ") }
      }
    */
    }
  }

  override def typeCheck(a: Any): Boolean = a == null || {
    a.isInstanceOf[Row] &&
      a.asInstanceOf[Row].toSeq.zip(fields).forall { case (v, f) =>
        val b = f.`type`.typeCheck(v)
        if (!b)
          println(s"v=$v, f=$f")
        b
      }
  }

  def schema = {
    assert(fields.length > 0)
    StructType(fields
      .map { case f =>
        StructField(f.name, f.`type`.schema)
      })
  }
}

object AST extends Positional {
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
    (g: (T) => Option[Any]): () => Any = {
    val f = subexpr.eval(c)
    () => {
      val x = f()
      if (x != null)
        g(x.asInstanceOf[T]).orNull
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

  def evalCompose[T1, T2, T3](c: EvalContext, subexpr1: AST, subexpr2: AST, subexpr3: AST)
    (g: (T1, T2, T3) => Any): () => Any = {
    val f1 = subexpr1.eval(c)
    val f2 = subexpr2.eval(c)
    val f3 = subexpr3.eval(c)
    () => {
      val x = f1()
      if (x != null) {
        val y = f2()
        if (y != null) {
          val z = f3()
          if (z != null)
            g(x.asInstanceOf[T1], y.asInstanceOf[T2], z.asInstanceOf[T3])
          else
            null
        } else
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
        g(convT.to(x))
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

case class Positioned[T](x: T) extends Positional

sealed abstract class AST(pos: Position, subexprs: Array[AST] = Array.empty) {
  var `type`: Type = null

  def this(posn: Position, subexpr1: AST) = this(posn, Array(subexpr1))

  def this(posn: Position, subexpr1: AST, subexpr2: AST) = this(posn, Array(subexpr1, subexpr2))

  def eval(c: EvalContext): () => Any

  def typecheckThis(typeSymTab: SymbolTable): Type = typecheckThis()

  def typecheckThis(): Type = throw new UnsupportedOperationException

  def typecheck(typeSymTab: SymbolTable) {
    subexprs.foreach(_.typecheck(typeSymTab))
    `type` = typecheckThis(typeSymTab)
  }

  def parseError(msg: String): Nothing = ParserUtils.error(pos, msg)
}

case class Const(posn: Position, value: Any, t: Type) extends AST(posn) {
  def eval(c: EvalContext): () => Any = {
    val v = value
    () => v
  }

  override def typecheckThis(): Type = t
}

case class Select(posn: Position, lhs: AST, rhs: String) extends AST(posn, lhs) {
  override def typecheckThis(): Type = {
    (lhs.`type`, rhs) match {
      case (TSample, "id") => TString
      case (TGenotype, "gt") => TInt
      case (TGenotype, "gtj") => TInt
      case (TGenotype, "gtk") => TInt
      case (TGenotype, "ad") => TArray(TInt)
      case (TGenotype, "dp") => TInt
      case (TGenotype, "od") => TInt
      case (TGenotype, "gq") => TInt
      case (TGenotype, "pl") => TArray(TInt)
      case (TGenotype, "isHomRef") => TBoolean
      case (TGenotype, "isHet") => TBoolean
      case (TGenotype, "isHomVar") => TBoolean
      case (TGenotype, "isCalledNonRef") => TBoolean
      case (TGenotype, "isHetNonRef") => TBoolean
      case (TGenotype, "isHetRef") => TBoolean
      case (TGenotype, "isCalled") => TBoolean
      case (TGenotype, "isNotCalled") => TBoolean
      case (TGenotype, "nNonRefAlleles") => TInt
      case (TGenotype, "pAB") => TDouble

      case (TVariant, "contig") => TString
      case (TVariant, "start") => TInt
      case (TVariant, "ref") => TString
      case (TVariant, "altAlleles") => TArray(TAltAllele)
      case (TVariant, "nAltAlleles") => TInt
      case (TVariant, "nAlleles") => TInt
      case (TVariant, "isBiallelic") => TBoolean
      case (TVariant, "nGenotypes") => TInt
      case (TVariant, "inParX") => TBoolean
      case (TVariant, "inParY") => TBoolean
      // assumes biallelic
      case (TVariant, "alt") => TString
      case (TVariant, "altAllele") => TAltAllele

      case (TAltAllele, "ref") => TString
      case (TAltAllele, "alt") => TString
      case (TAltAllele, "isSNP") => TBoolean
      case (TAltAllele, "isMNP") => TBoolean
      case (TAltAllele, "isIndel") => TBoolean
      case (TAltAllele, "isInsertion") => TBoolean
      case (TAltAllele, "isDeletion") => TBoolean
      case (TAltAllele, "isComplex") => TBoolean
      case (TAltAllele, "isTransition") => TBoolean
      case (TAltAllele, "isTransversion") => TBoolean

      case (t: TStruct, _) => {
        t.selfField(rhs) match {
          case Some(f) => f.`type`
          case None => parseError(s"`$t' has no field `$rhs")
        }
      }
      case (t: TNumeric, "toInt") => TInt
      case (t: TNumeric, "toLong") => TLong
      case (t: TNumeric, "toFloat") => TFloat
      case (t: TNumeric, "toDouble") => TDouble
      case (t: TNumeric, "abs") => t
      case (t: TNumeric, "signum") => TInt
      case (TString, "length") => TInt
      case (TArray(_), "length") => TInt
      case (TArray(_), "isEmpty") => TBoolean
      case (TSet(_), "size") => TInt
      case (TSet(_), "isEmpty") => TBoolean

      case (_, "isMissing") => TBoolean
      case (_, "isNotMissing") => TBoolean

      case (t, _) =>
        parseError(s"`$t' has no field `$rhs'")
    }
  }

  def eval(c: EvalContext): () => Any = ((lhs.`type`, rhs): @unchecked) match {
    case (TSample, "id") => lhs.eval(c)

    case (TGenotype, "gt") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.gt)
    case (TGenotype, "gtj") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.gt.map(gtx => Genotype.gtPair(gtx).j))
    case (TGenotype, "gtk") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.gt.map(gtx => Genotype.gtPair(gtx).k))
    case (TGenotype, "ad") =>
      AST.evalFlatCompose[Genotype](c, lhs)(g => g.ad.map(a => a: IndexedSeq[Int]))
    case (TGenotype, "dp") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.dp)
    case (TGenotype, "od") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.od)
    case (TGenotype, "gq") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.gq)
    case (TGenotype, "pl") =>
      AST.evalFlatCompose[Genotype](c, lhs)(g => g.pl.map(a => a: IndexedSeq[Int]))
    case (TGenotype, "isHomRef") =>
      AST.evalCompose[Genotype](c, lhs)(_.isHomRef)
    case (TGenotype, "isHet") =>
      AST.evalCompose[Genotype](c, lhs)(_.isHet)
    case (TGenotype, "isHomVar") =>
      AST.evalCompose[Genotype](c, lhs)(_.isHomVar)
    case (TGenotype, "isCalledNonRef") =>
      AST.evalCompose[Genotype](c, lhs)(_.isCalledNonRef)
    case (TGenotype, "isHetNonRef") =>
      AST.evalCompose[Genotype](c, lhs)(_.isHetNonRef)
    case (TGenotype, "isHetRef") =>
      AST.evalCompose[Genotype](c, lhs)(_.isHetRef)
    case (TGenotype, "isCalled") =>
      AST.evalCompose[Genotype](c, lhs)(_.isCalled)
    case (TGenotype, "isNotCalled") =>
      AST.evalCompose[Genotype](c, lhs)(_.isNotCalled)
    case (TGenotype, "nNonRefAlleles") => AST.evalFlatCompose[Genotype](c, lhs)(_.nNonRefAlleles)
    case (TGenotype, "pAB") =>
      AST.evalFlatCompose[Genotype](c, lhs)(_.pAB())

    case (TVariant, "contig") =>
      AST.evalCompose[Variant](c, lhs)(_.contig)
    case (TVariant, "start") =>
      AST.evalCompose[Variant](c, lhs)(_.start)
    case (TVariant, "ref") =>
      AST.evalCompose[Variant](c, lhs)(_.ref)
    case (TVariant, "altAlleles") =>
      AST.evalCompose[Variant](c, lhs)(_.altAlleles)
    case (TVariant, "nAltAlleles") =>
      AST.evalCompose[Variant](c, lhs)(_.nAltAlleles)
    case (TVariant, "nAlleles") =>
      AST.evalCompose[Variant](c, lhs)(_.nAlleles)
    case (TVariant, "isBiallelic") =>
      AST.evalCompose[Variant](c, lhs)(_.isBiallelic)
    case (TVariant, "nGenotypes") =>
      AST.evalCompose[Variant](c, lhs)(_.nGenotypes)
    case (TVariant, "inParX") =>
      AST.evalCompose[Variant](c, lhs)(_.inParX)
    case (TVariant, "inParY") =>
      AST.evalCompose[Variant](c, lhs)(_.inParY)
    // assumes biallelic
    case (TVariant, "alt") =>
      AST.evalCompose[Variant](c, lhs)(_.alt)
    case (TVariant, "altAllele") =>
      AST.evalCompose[Variant](c, lhs)(_.altAllele)

    case (TAltAllele, "ref") => AST.evalCompose[AltAllele](c, lhs)(_.ref)
    case (TAltAllele, "alt") => AST.evalCompose[AltAllele](c, lhs)(_.alt)
    case (TAltAllele, "isSNP") => AST.evalCompose[AltAllele](c, lhs)(_.isSNP)
    case (TAltAllele, "isMNP") => AST.evalCompose[AltAllele](c, lhs)(_.isMNP)
    case (TAltAllele, "isIndel") => AST.evalCompose[AltAllele](c, lhs)(_.isIndel)
    case (TAltAllele, "isInsertion") => AST.evalCompose[AltAllele](c, lhs)(_.isInsertion)
    case (TAltAllele, "isDeletion") => AST.evalCompose[AltAllele](c, lhs)(_.isDeletion)
    case (TAltAllele, "isComplex") => AST.evalCompose[AltAllele](c, lhs)(_.isComplex)
    case (TAltAllele, "isTransition") => AST.evalCompose[AltAllele](c, lhs)(_.isTransition)
    case (TAltAllele, "isTransversion") => AST.evalCompose[AltAllele](c, lhs)(_.isTransversion)

    case (t: TStruct, _) =>
      val Some(f) = t.selfField(rhs)
      val i = f.index
      AST.evalCompose[Row](c, lhs)(_.get(i))

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

    case (_, "isMissing") =>
      val f = lhs.eval(c)
      () => f() == null
    case (_, "isNotMissing") =>
      val f = lhs.eval(c)
      () => f() != null

    case (TInt, "abs") => AST.evalCompose[Int](c, lhs)(_.abs)
    case (TLong, "abs") => AST.evalCompose[Long](c, lhs)(_.abs)
    case (TFloat, "abs") => AST.evalCompose[Float](c, lhs)(_.abs)
    case (TDouble, "abs") => AST.evalCompose[Double](c, lhs)(_.abs)

    case (TInt, "signum") => AST.evalCompose[Int](c, lhs)(_.signum)
    case (TLong, "signum") => AST.evalCompose[Long](c, lhs)(_.signum)
    case (TFloat, "signum") => AST.evalCompose[Float](c, lhs)(_.signum)
    case (TDouble, "signum") => AST.evalCompose[Double](c, lhs)(_.signum)

    case (TString, "length") => AST.evalCompose[String](c, lhs)(_.length)

    case (TArray(_), "length") => AST.evalCompose[IndexedSeq[_]](c, lhs)(_.length)
    case (TArray(_), "isEmpty") => AST.evalCompose[IndexedSeq[_]](c, lhs)(_.isEmpty)

    case (TSet(_), "size") => AST.evalCompose[IndexedSeq[_]](c, lhs)(_.size)
    case (TSet(_), "isEmpty") => AST.evalCompose[IndexedSeq[_]](c, lhs)(_.isEmpty)
  }

}

case class Lambda(posn: Position, param: String, body: AST) extends AST(posn, body) {
  def typecheck(): Type = parseError("non-function context")

  def eval(c: EvalContext): () => Any = throw new UnsupportedOperationException
}

case class ApplyMethod(posn: Position, lhs: AST, method: String, args: Array[AST]) extends AST(posn, lhs +: args) {
  override def typecheck(symTab: SymbolTable) {
    (method, args) match {
      case ("find", Array(Lambda(_, param, body))) =>
        lhs.typecheck(symTab)

        val elementType = lhs.`type` match {
          case TArray(elementType) => elementType
          case _ =>
            fatal("no `$method' on non-array")
        }

        `type` = elementType

        // index unused in typecheck
        body.typecheck(symTab + (param ->(-1, elementType)))
        if (body.`type` != TBoolean)
          fatal(s"expected Boolean, got `${body.`type`}' in first argument to `$method'")

      case _ =>
        super.typecheck(symTab)
    }
  }

  override def typecheckThis(): Type = {
    (lhs.`type`, method, args.map(_.`type`)) match {
      case (TArray(elementType), "contains", Array(TString)) => TBoolean
      case (TArray(TString), "mkString", Array(TString)) => TString
      case (TSet(elementType), "contains", Array(TString)) => TBoolean
      case (TString, "split", Array(TString)) => TArray(TString)

      case (t: TNumeric, "min", Array(t2: TNumeric)) =>
        AST.promoteNumeric(t, t2)
      case (t: TNumeric, "max", Array(t2: TNumeric)) =>
        AST.promoteNumeric(t, t2)

      case (t, "orElse", Array(t2)) if t == t2 =>
        t

      case (t, _, _) =>
        parseError(s"`no matching signature for `$method' on `$t'")
    }
  }

  def eval(c: EvalContext): () => Any = ((lhs.`type`, method, args): @unchecked) match {
    case (returnType, "find", Array(Lambda(_, param, body))) =>
      val localIdx = c.a.length
      c.a += null
      val bodyFn = body.eval(c.copy(
        symTab = c.symTab + (param ->(localIdx, returnType))))
      val localA = c.a
      AST.evalCompose[IndexedSeq[_]](c, lhs) { case is =>
        def f(i: Int): Any =
          if (i < is.length) {
            val elt = is(i)
            localA(localIdx) = elt
            val r = bodyFn()
            if (r != null
              && r.asInstanceOf[Boolean])
              elt
            else
              f(i + 1)
          } else
            null
        f(0)
      }

    case (_, "orElse", Array(a)) =>
      val f1 = lhs.eval(c)
      val f2 = a.eval(c)
      () => {
        val v = f1()
        if (v == null)
          f2()
        else
          v
      }

    case (TArray(elementType), "contains", Array(a)) =>
      AST.evalCompose[IndexedSeq[_], Any](c, lhs, a) { case (a, x) => a.contains(x) }
    case (TArray(TString), "mkString", Array(a)) =>
      AST.evalCompose[IndexedSeq[String], String](c, lhs, a) { case (s, t) => s.mkString(t) }
    case (TSet(elementType), "contains", Array(a)) =>
      AST.evalCompose[IndexedSeq[Any], Any](c, lhs, a) { case (a, x) => a.contains(x) }

    case (TString, "split", Array(a)) =>
      AST.evalCompose[String, String](c, lhs, a) { case (s, p) => s.split(p): IndexedSeq[String] }

    case (TInt, "min", Array(a)) => AST.evalComposeNumeric[Int, Int](c, lhs, a)(_ min _)
    case (TLong, "min", Array(a)) => AST.evalComposeNumeric[Long, Long](c, lhs, a)(_ min _)
    case (TFloat, "min", Array(a)) => AST.evalComposeNumeric[Float, Float](c, lhs, a)(_ min _)
    case (TDouble, "min", Array(a)) => AST.evalComposeNumeric[Double, Double](c, lhs, a)(_ min _)

    case (TInt, "max", Array(a)) => AST.evalComposeNumeric[Int, Int](c, lhs, a)(_ max _)
    case (TLong, "max", Array(a)) => AST.evalComposeNumeric[Long, Long](c, lhs, a)(_ max _)
    case (TFloat, "max", Array(a)) => AST.evalComposeNumeric[Float, Float](c, lhs, a)(_ max _)
    case (TDouble, "max", Array(a)) => AST.evalComposeNumeric[Double, Double](c, lhs, a)(_ max _)
  }
}

case class BinaryOp(posn: Position, lhs: AST, operation: String, rhs: AST) extends AST(posn, lhs, rhs) {
  def eval(c: EvalContext): () => Any = ((operation, `type`): @unchecked) match {
    case ("+", TString) => AST.evalCompose[String, String](c, lhs, rhs)(_ + _)
    case ("~", TBoolean) => AST.evalCompose[String, String](c, lhs, rhs) { (s, t) =>
      s.r.findFirstIn(t).isDefined
    }

    case ("||", TBoolean) => {
      val f1 = lhs.eval(c)
      val f2 = rhs.eval(c)

      () => {
        val x = f1()
        if (x != null) {
          if (x.asInstanceOf[Boolean])
            true
          else
            f2()
        } else {
          val x2 = f2()
          if (x != null
            && x.asInstanceOf[Boolean])
            true
          else
            null
        }
      }
    }

    case ("&&", TBoolean) => {
      val f1 = lhs.eval(c)
      val f2 = rhs.eval(c)
      () => {
        val x = f1()
        if (x != null) {
          if (x.asInstanceOf[Boolean])
            f2()
          else
            false
        } else {
          val x2 = f2()
          if (x2 != null
            && !x2.asInstanceOf[Boolean])
            false
          else
            null
        }
      }
    }

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
    case (TString, "~", TString) => TBoolean
    case (TBoolean, "||", TBoolean) => TBoolean
    case (TBoolean, "&&", TBoolean) => TBoolean
    case (lhsType: TNumeric, "+", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "-", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "*", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "/", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)

    case (lhsType, _, rhsType) =>
      parseError(s"invalid arguments to `$operation': ($lhsType, $rhsType)")
  }
}

case class Comparison(posn: Position, lhs: AST, operation: String, rhs: AST) extends AST(posn, lhs, rhs) {
  var operandType: Type = null

  def eval(c: EvalContext): () => Any = ((operation, operandType): @unchecked) match {
    case ("==", _) => AST.evalCompose[Any, Any](c, lhs, rhs)(_ == _)
    case ("!=", _) => AST.evalCompose[Any, Any](c, lhs, rhs)(_ != _)

    case ("<", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ < _)
    case ("<=", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ <= _)
    case (">", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ > _)
    case (">=", TInt) => AST.evalComposeNumeric[Int, Int](c, lhs, rhs)(_ >= _)

    case ("<", TLong) => AST.evalComposeNumeric[Long, Long](c, lhs, rhs)(_ < _)
    case ("<=", TLong) => AST.evalComposeNumeric[Long, Long](c, lhs, rhs)(_ <= _)
    case (">", TLong) => AST.evalComposeNumeric[Long, Long](c, lhs, rhs)(_ > _)
    case (">=", TLong) => AST.evalComposeNumeric[Long, Long](c, lhs, rhs)(_ >= _)

    case ("<", TFloat) => AST.evalComposeNumeric[Float, Float](c, lhs, rhs)(_ < _)
    case ("<=", TFloat) => AST.evalComposeNumeric[Float, Float](c, lhs, rhs)(_ <= _)
    case (">", TFloat) => AST.evalComposeNumeric[Float, Float](c, lhs, rhs)(_ > _)
    case (">=", TFloat) => AST.evalComposeNumeric[Float, Float](c, lhs, rhs)(_ >= _)

    case ("<", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ < _)
    case ("<=", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ <= _)
    case (">", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ > _)
    case (">=", TDouble) => AST.evalComposeNumeric[Double, Double](c, lhs, rhs)(_ >= _)
  }

  override def typecheckThis(): Type = {
    operandType = (lhs.`type`, operation, rhs.`type`) match {
      case (_, "==" | "!=", _) => null
      case (lhsType: TNumeric, "<=" | ">=" | "<" | ">", rhsType: TNumeric) =>
        AST.promoteNumeric(lhsType, rhsType)

      case (lhsType, _, rhsType) =>
        parseError(s"invalid arguments to `$operation': ($lhsType, $rhsType)")
    }

    TBoolean
  }
}

case class UnaryOp(posn: Position, operation: String, operand: AST) extends AST(posn, operand) {
  def eval(c: EvalContext): () => Any = ((operation, `type`): @unchecked) match {
    case ("-", TInt) => AST.evalComposeNumeric[Int](c, operand)(-_)
    case ("-", TLong) => AST.evalComposeNumeric[Long](c, operand)(-_)
    case ("-", TFloat) => AST.evalComposeNumeric[Float](c, operand)(-_)
    case ("-", TDouble) => AST.evalComposeNumeric[Double](c, operand)(-_)

    case ("!", TBoolean) => AST.evalCompose[Boolean](c, operand)(!_)
  }

  override def typecheckThis(): Type = (operation, operand.`type`) match {
    case ("-", t: TNumeric) => AST.promoteNumeric(t)
    case ("!", TBoolean) => TBoolean

    case (_, t) =>
      parseError(s"invalid argument to unary `$operation': ${t.toString}")
  }
}

case class IndexArray(posn: Position, f: AST, idx: AST) extends AST(posn, Array(f, idx)) {
  override def typecheckThis(): Type = (f.`type`, idx.`type`) match {
    case (TArray(elementType), TInt) => elementType
    case (TString, TInt) => TChar

    case _ =>
      parseError("invalid array index expression")
  }

  def eval(c: EvalContext): () => Any = ((f.`type`, idx.`type`): @unchecked) match {
    case (TArray(elementType), TInt) =>
      AST.evalCompose[IndexedSeq[_], Int](c, f, idx)((a, i) => a(i))

    case (TString, TInt) =>
      AST.evalCompose[String, Int](c, f, idx)((s, i) => s(i))
  }

}

case class SymRef(posn: Position, symbol: String) extends AST(posn) {
  def eval(c: EvalContext): () => Any = {
    val i = c.symTab(symbol)._1
    val localA = c.a
    () => localA(i)
  }

  override def typecheckThis(typeSymTab: SymbolTable): Type = typeSymTab.get(symbol) match {
    case Some((_, t)) => t
    case None =>
      parseError(s"symbol `$symbol' not found")
  }
}

case class If(pos: Position, cond: AST, thenTree: AST, elseTree: AST)
  extends AST(pos, Array(cond, thenTree, elseTree)) {
  override def typecheckThis(typeSymTab: SymbolTable): Type =
    TBoolean

  def eval(c: EvalContext): () => Any = {
    val f1 = cond.eval(c)
    val f2 = thenTree.eval(c)
    val f3 = elseTree.eval(c)
    () => {
      val c = f1()
      if (c != null) {
        if (c.asInstanceOf[Boolean])
          f2()
        else
          f3()
      } else
        null
    }
  }
}
