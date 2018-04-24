package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.asm4s.coerce

import scala.collection.mutable

object IRFunctionRegistry {

  val irRegistry: mutable.MultiMap[String, (Seq[Type], Seq[IR] => IR)] =
    new mutable.HashMap[String, mutable.Set[(Seq[Type], Seq[IR] => IR)]] with mutable.MultiMap[String, (Seq[Type], Seq[IR] => IR)]

  val codeRegistry: mutable.MultiMap[String, (Seq[Type], IRFunction)] =
    new mutable.HashMap[String, mutable.Set[(Seq[Type], IRFunction)]] with mutable.MultiMap[String, (Seq[Type], IRFunction)]

  def addIRFunction(f: IRFunction): Unit =
    codeRegistry.addBinding(f.name, (f.argTypes, f))

  def addIR(name: String, types: Seq[Type], f: Seq[IR] => IR): Unit =
    irRegistry.addBinding(name, (types, f))

  private def lookupInRegistry[T](reg: mutable.MultiMap[String, (Seq[Type], T)], name: String, args: Seq[Type]): Option[T] = {
    reg.lift(name).flatMap { fs =>
      fs.filter { case (ts, _) =>
        ts.length == args.length && {
          ts.foreach(_.clear())
          (ts, args).zipped.forall(_.unify(_))
        }
      }.toSeq match {
        case Seq() => None
        case Seq((_, f)) => Some(f)
        case _ => fatal(s"Multiple functions found that satisfy $name(${ args.mkString(",") }).")
      }
    }
  }

  def lookupFunction(name: String, args: Seq[Type]): Option[IRFunction] =
    lookupInRegistry(codeRegistry, name, args)

  def lookupConversion(name: String, args: Seq[Type]): Option[Seq[IR] => IR] = {
    assert(args.forall(_ != null))

    val validIR = lookupInRegistry(irRegistry, name, args)

    val validMethods = lookupFunction(name, args).map { f =>
      { irArgs: Seq[IR] =>
        f match {
          case irf: IRFunctionWithoutMissingness => Apply(name, irArgs)
          case irf: IRFunctionWithMissingness => ApplySpecial(name, irArgs)
        }
      }
    }

    (validIR, validMethods) match {
      case (None, None) =>
        log.warn(s"no IRFunction found for $name(${ args.mkString(", ") })")
        None
      case (None, Some(x)) => Some(x)
      case (Some(x), None) => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name(${ args.mkString(",") }).")
    }
  }

  CallFunctions.registerAll()
  GenotypeFunctions.registerAll()
  MathFunctions.registerAll()
  UtilFunctions.registerAll()
  StringFunctions.registerAll()
}

abstract class RegistryFunctions {

  def registerAll(): Unit

  private val boxes = mutable.Map[String, Box[Type]]()

  private def tvBoxes(name: String) = boxes.getOrElseUpdate(name, Box[Type]())

  def tv(name: String): TVariable =
    TVariable(name, b = tvBoxes(name))

  def tv(name: String, cond: Type => Boolean): TVariable =
    TVariable(name, cond, tvBoxes(name))

  def tnum(name: String): TVariable =
    tv(name, _.isInstanceOf[TNumeric])

  def registerCode(mname: String, aTypes: Array[Type], rType: Type, isDet: Boolean)(impl: (EmitMethodBuilder, Array[Code[_]]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new IRFunctionWithoutMissingness {
      val isDeterministic: Boolean = isDet

      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def apply(mb: EmitMethodBuilder, args: Code[_]*): Code[_] = impl(mb, args.toArray)
    })
  }

  def registerCodeWithMissingness(mname: String, aTypes: Array[Type], rType: Type, isDet: Boolean)(impl: (EmitMethodBuilder, Array[EmitTriplet]) => EmitTriplet) {
    IRFunctionRegistry.addIRFunction(new IRFunctionWithMissingness {
      val isDeterministic: Boolean = isDet

      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet = impl(mb, args.toArray)
    })
  }

  def registerScalaFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String, isDeterministic: Boolean) {
    registerCode(mname, argTypes, rType, isDeterministic) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeScalaObject(cls, method, cts, args)(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerScalaFunction(mname: String, types: Type*)(cls: Class[_], method: String, isDeterministic: Boolean = true): Unit =
    registerScalaFunction(mname: String, types.init.toArray, types.last)(cls, method, isDeterministic)

  def registerJavaStaticFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String, isDeterministic: Boolean) {
    registerCode(mname, argTypes, rType, isDeterministic) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic(cls, method, cts, args)(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerJavaStaticFunction(mname: String, types: Type*)(cls: Class[_], method: String, isDeterministic: Boolean = true): Unit =
    registerJavaStaticFunction(mname, types.init.toArray, types.last)(cls, method, isDeterministic)

  def registerIR(mname: String, argTypes: Array[Type])(f: Seq[IR] => IR) {
    IRFunctionRegistry.addIR(mname, argTypes, f)
  }

  def registerCode(mname: String, rt: Type, isDeterministic: Boolean = true)(impl: EmitMethodBuilder => Code[_]): Unit =
    registerCode(mname, Array[Type](), rt, isDeterministic) { case (mb, Array()) => impl(mb) }

  def registerCode[T1: TypeInfo](mname: String, mt1: Type, rt: Type, isDeterministic: Boolean = true)(impl: (EmitMethodBuilder, Code[T1]) => Code[_]): Unit =
    registerCode(mname, Array(mt1), rt, isDeterministic) { case (mb, Array(a1)) => impl(mb, coerce[T1](a1)) }

  def registerCode[T1: TypeInfo, T2: TypeInfo](mname: String, mt1: Type, mt2: Type, rt: Type, isDeterministic: Boolean = true)(impl: (EmitMethodBuilder, Code[T1], Code[T2]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2), rt, isDeterministic) { case (mb, Array(a1, a2)) => impl(mb, coerce[T1](a1), coerce[T2](a2)) }

  def registerCode[T1: TypeInfo, T2: TypeInfo, T3: TypeInfo](mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, isDeterministic: Boolean = true)
    (impl: (EmitMethodBuilder, Code[_], Code[_], Code[_]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3), r, isDeterministic) { case (mb, Array(a1, a2, a3)) => impl(mb, coerce[T1](a1), coerce[T2](a2), coerce[T3](a3)) }

  def registerCodeWithMissingness(mname: String, rt: Type, isDeterministic: Boolean = true)(impl: EmitMethodBuilder => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array[Type](), rt, isDeterministic) { case (mb, Array()) => impl(mb) }

  def registerCodeWithMissingness(mname: String, mt1: Type, rt: Type, isDeterministic: Boolean = true)(impl: (EmitMethodBuilder, EmitTriplet) => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array(mt1), rt, isDeterministic) { case (mb, Array(a1)) => impl(mb, a1) }

  def registerCodeWithMissingness(mname: String, mt1: Type, mt2: Type, rt: Type, isDeterministic: Boolean = true)(impl: (EmitMethodBuilder, EmitTriplet, EmitTriplet) => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array(mt1, mt2), rt, isDeterministic) { case (mb, Array(a1, a2)) => impl(mb, a1, a2) }

  def registerIR(mname: String)(f: () => IR): Unit =
    registerIR(mname, Array[Type]()) { case Seq() => f() }

  def registerIR(mname: String, mt1: Type)(f: IR => IR): Unit =
    registerIR(mname, Array(mt1)) { case Seq(a1) => f(a1) }

  def registerIR(mname: String, mt1: Type, mt2: Type)(f: (IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2)) { case Seq(a1, a2) => f(a1, a2) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type)(f: (IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3)) { case Seq(a1, a2, a3) => f(a1, a2, a3) }
}

sealed abstract class IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet

  def getAsMethod(fb: EmitFunctionBuilder[_], args: Type*): EmitMethodBuilder = ???

  def returnType: Type

  override def toString: String = s"$name(${ argTypes.mkString(", ") }): $returnType"

  def isDeterministic: Boolean

}

abstract class IRFunctionWithoutMissingness extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: EmitMethodBuilder, args: Code[_]*): Code[_]

  def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet = {
    val setup = args.map(_.setup)
    val missing = args.map(_.m).reduce(_ || _)
    val value = apply(mb, args.map(_.v): _*)

    EmitTriplet(setup, missing, value)
  }

  override def getAsMethod(fb: EmitFunctionBuilder[_], args: Type*): EmitMethodBuilder = {
    argTypes.foreach(_.clear())
    (argTypes, args).zipped.foreach(_.unify(_))
    val ts = argTypes.map(t => typeToTypeInfo(t.subst()))
    val methodbuilder = fb.newMethod((typeInfo[Region] +: ts).toArray, typeToTypeInfo(returnType))
    methodbuilder.emit(apply(methodbuilder, ts.zipWithIndex.map { case (a, i) => methodbuilder.getArg(i + 2)(a).load() }: _*))
    methodbuilder
  }

  def returnType: Type

  override def toString: String = s"$name(${ argTypes.mkString(", ") }): $returnType"
}

abstract class IRFunctionWithMissingness extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet

  def returnType: Type

  override def toString: String = s"$name(${ argTypes.mkString(", ") }): $returnType"
}
