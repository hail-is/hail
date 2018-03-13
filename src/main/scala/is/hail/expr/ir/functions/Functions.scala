package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.{AST, FunType, Lambda, TypeTag}
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._

import scala.collection.mutable
import scala.reflect.ClassTag

object IRFunctionRegistry {

  val registry: mutable.Map[String, Seq[(TypeTag, Seq[IR] => IR)]] = mutable.Map().withDefaultValue(Seq.empty)

  UtilFunctions.registerAll(registry)

  def lookupFunction(name: String, args: TypeTag): Option[Seq[IR] => IR] = {
    val validMethods = registry(name).flatMap { case (tt, f) =>
      if (tt.xs.length == args.xs.length) {
        tt.clear()
        if (tt.unify(args))
          Some(f)
        else
          None
      }
      else {
        None
      }
    }
    validMethods match {
      case Seq() => None
      case Seq(x) => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name$args.")
    }
  }
}

abstract class RegistryFunctions {

  def registerAll(r: mutable.Map[String, Seq[(TypeTag, Seq[IR] => IR)]]) {
    registry.foreach { case (k, v) =>
      r.put(k, r(k) ++ registry(k))
    }
  }

  private val registry = mutable.Map[String, Seq[(TypeTag, Seq[IR] => IR)]]().withDefaultValue(Seq.empty)

  private var tvs = mutable.Map[String, TVariable]()

  def tv(name: String): TVariable = {
    tvs.getOrElse(name, {
      tvs.put(name, TVariable(name))
      tvs(name)
    })
  }

  def addIRFunction(f: IRFunction) {
    val l = registry(f.name)
    registry.put(f.name,
      l :+ (f.types, { args: Seq[IR] =>
        ApplyFunction(f, args)
      }))
  }

  def addIR(name: String, types: Seq[Type], f: Seq[IR] => IR) {
    val l = registry(name)
    registry.put(name, l :+ ((FunType(types: _*), f)))
  }

  def registerCode[R](mname: String, mtypes: Array[Type])(impl: (MethodBuilder, Array[Code[_]]) => Code[R]) {
    addIRFunction(new IRFunction {
      override val name: String = mname

      override val types: TypeTag = FunType(mtypes.init: _*)

      override val returnType: Type = mtypes.last

      override def apply(mb: MethodBuilder, args: Code[_]*): Code[R] = impl(mb, args.toArray)
    })
  }

  def registerScalaFunction[R: ClassTag](mname: String, types: Type*)(obj: Any, method: String)
  {
    registerCode[R](mname, types.toArray)
      { (mb, args) =>
        val cts = types.init.map(TypeToIRIntermediateClassTag(_).runtimeClass).toArray
        Code.invokeScalaObject[R](obj, method, cts, args)
      }
  }

  def registerJavaStaticFunction[C: ClassTag, R: ClassTag](mname: String, types: Type*)(method: String) {
    registerCode[R](mname, types.toArray)
      { (mb, args) =>
        val cts = types.init.map(TypeToIRIntermediateClassTag(_).runtimeClass)
        Code.invokeStatic[C, R](method, cts.toArray, args)
      }
  }

  def registerIR(mname: String, types: Array[Type])(f: Seq[IR] => IR) {
    addIR(mname, types.init, f)
  }

  def registerCode[R](mname: String, rt: Type)(impl: MethodBuilder => Code[R]): Unit =
    registerCode[R](mname, Array(rt)) { case (mb, Array()) => impl(mb) }

  def registerCode[R](mname: String, mt1: Type, rt: Type)(impl: (MethodBuilder, Code[_]) => Code[R]): Unit =
    registerCode[R](mname, Array(mt1, rt)) { case (mb, Array(a1)) => impl(mb, a1) }

  def registerCode[R](mname: String, mt1: Type, mt2: Type, rt: Type)(impl: (MethodBuilder, Code[_], Code[_]) => Code[R]): Unit =
    registerCode[R](mname, Array(mt1, mt2, rt)) { case (mb, Array(a1, a2)) => impl(mb, a1, a2) }

  def registerCode[R](mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type)(impl: (MethodBuilder, Code[_], Code[_], Code[_]) => Code[R]): Unit =
    registerCode[R](mname, Array(mt1, mt2, mt3, rt)) { case (mb, Array(a1, a2, a3)) => impl(mb, a1, a2, a3) }

  def registerIR(mname: String, rt: Type)(f: () => IR): Unit =
    registerIR(mname, Array(rt)){ case Seq() => f() }

  def registerIR(mname: String, mt1: Type, rt: Type)(f: IR => IR): Unit =
    registerIR(mname, Array(mt1, rt)){ case Seq(a1) => f(a1) }

  def registerIR(mname: String, mt1: Type, mt2: Type, rt: Type)(f: (IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, rt)){ case Seq(a1, a2) => f(a1, a2) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type)(f: (IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3, rt)){ case Seq(a1, a2, a3) => f(a1, a2, a3) }

}

object IRFromAST {

  private def tryPrimOpConversion(fn: String): IndexedSeq[IR] => Option[IR] =
    flatLift {
      case IndexedSeq(x) => for {
        op <- UnaryOp.fromString.lift(fn)
        t <- UnaryOp.returnTypeOption(op, x.typ)
      } yield ApplyUnaryPrimOp(op, x, t)
      case IndexedSeq(x, y) => for {
        op <- BinaryOp.fromString.lift(fn)
        t <- BinaryOp.returnTypeOption(op, x.typ, y.typ)
      } yield ApplyBinaryPrimOp(op, x, y, t)
    }

  def lookup(name: String, args: IndexedSeq[IR], types: IndexedSeq[Type]): Option[IR] = {
    (name, args) match {
      case ("isDefined", IndexedSeq(x)) => Some(ApplyUnaryPrimOp(Bang(), IsNA(x)))
      case ("orMissing", IndexedSeq(cond, x)) => Some(If(cond, x, NA(x.typ)))
      case ("size", IndexedSeq(x)) if x.typ.isInstanceOf[TArray] => Some(ArrayLen(x))

      case (n, a) =>
        tryPrimOpConversion(name)(a).orElse(
          IRFunctionRegistry.lookupFunction(n, FunType(types: _*))
            .map { irf => irf(a) })
    }
  }

  def convertLambda(method: String, lhs: AST, lambda: Lambda, agg: Option[String]): Option[IR] = {
    (lhs.`type`, method) match {
      case (t: TArray, "map") =>
        for {
          ir <- lhs.toIR(agg)
          body <- lambda.body.toIR(agg)
        } yield {
          ArrayMap(ir, lambda.param, body, t.elementType)
        }
      case _ => None
    }
  }
}

abstract class IRFunction {
  def name: String

  def types: TypeTag

  def apply(mb: MethodBuilder, args: Code[_]*): Code[_]

  def returnType: Type

}