package is.hail.expr.ir.functions

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.asm4s.coerce
import is.hail.experimental.ExperimentalFunctions
import is.hail.expr.types.physical.{PString, PType}
import is.hail.expr.types.virtual._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect._

object IRFunctionRegistry {

  val irRegistry: mutable.MultiMap[String, (Seq[Type], Type, Seq[IR] => IR)] =
    new mutable.HashMap[String, mutable.Set[(Seq[Type], Type, Seq[IR] => IR)]] with mutable.MultiMap[String, (Seq[Type], Type, Seq[IR] => IR)]

  val codeRegistry: mutable.MultiMap[String, IRFunction] =
    new mutable.HashMap[String, mutable.Set[IRFunction]] with mutable.MultiMap[String, IRFunction]

  def addIRFunction(f: IRFunction): Unit =
    codeRegistry.addBinding(f.name, f)

  def addIR(name: String, argTypes: Seq[Type], retType: Type, f: Seq[IR] => IR): Unit =
    irRegistry.addBinding(name, (argTypes, retType, f))

  def pyRegisterIR(mname: String,
    argNames: java.util.ArrayList[String],
    argTypes: java.util.ArrayList[String], retType: String,
    body: IR): Unit = {
    addIR(mname,
      argTypes.asScala.toArray.map(IRParser.parseType), IRParser.parseType(retType), { args =>
        Subst(body,
          Env[IR](argNames.asScala.zip(args): _*))
      })
  }

  def removeIRFunction(name: String, args: Seq[Type]): Unit = {
    val functions = codeRegistry(name)
    val toRemove = functions.filter(_.unify(args)).toArray
    assert(toRemove.length == 1)
    codeRegistry.removeBinding(name, toRemove.head)
  }

  def removeIRFunction(name: String): Unit =
    codeRegistry.remove(name)

  private def lookupInRegistry[T](reg: mutable.MultiMap[String, T], name: String, args: Seq[Type], cond: (T, Seq[Type]) => Boolean): Option[T] = {
    reg.lift(name).map { fs => fs.filter(t => cond(t, args)).toSeq }.getOrElse(FastSeq()) match {
      case Seq() => None
      case Seq(f) => Some(f)
      case _ => fatal(s"Multiple functions found that satisfy $name(${ args.mkString(",") }).")
    }
  }

  def lookupFunction(name: String, args: Seq[Type]): Option[IRFunction] =
    lookupInRegistry(codeRegistry, name, args, (f: IRFunction, ts: Seq[Type]) => f.unify(ts))

  def lookupConversion(name: String, args: Seq[Type]): Option[Seq[IR] => IR] = {
    type Conversion = (Seq[Type], Type, Seq[IR] => IR)
    val findIR: (Conversion, Seq[Type]) => Boolean = {
      case ((ts, _, _), t2s) =>
        ts.length == args.length && {
          ts.foreach(_.clear())
          (ts, t2s).zipped.forall(_.unify(_))
        }
    }
    val validIR: Option[Seq[IR] => IR] = lookupInRegistry[Conversion](irRegistry, name, args, findIR).map {
      case (_, _, conversion) => args =>
        val x = ApplyIR(name, args)
        x.conversion = conversion
        x
    }

    val validMethods = lookupFunction(name, args).map { f => { irArgs: Seq[IR] =>
      f match {
        case _: SeededIRFunction =>
          ApplySeeded(name, irArgs.init, irArgs.last.asInstanceOf[I64].x)
        case _: IRFunctionWithoutMissingness => Apply(name, irArgs)
        case _: IRFunctionWithMissingness => ApplySpecial(name, irArgs)
      }
    }
    }

    (validIR, validMethods) match {
      case (None, None) =>
        None
      case (None, Some(x)) => Some(x)
      case (Some(x), None) => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name(${ args.mkString(",") }).")
    }
  }

  Seq(
    ArrayFunctions,
    CallFunctions,
    DictFunctions,
    GenotypeFunctions,
    IntervalFunctions,
    LocusFunctions,
    MathFunctions,
    RandomSeededFunctions,
    SetFunctions,
    StringFunctions,
    UtilFunctions,
    ExperimentalFunctions
  ).foreach(_.registerAll())

  def dumpFunctions(): Unit = {
    def dtype(t: Type): String = s"""dtype("${ StringEscapeUtils.escapeString(t.toString) }\")"""

    irRegistry.foreach { case (name, fns) =>
        fns.foreach { case (argTypes, retType, f) =>
          println(s"""register_function("${ StringEscapeUtils.escapeString(name) }", (${ argTypes.map(dtype).mkString(",") }), ${ dtype(retType) })""")
        }
    }

    codeRegistry.foreach { case (name, fns) =>
        fns.foreach { f =>
          println(s"""${
            if (f.isInstanceOf[SeededIRFunction])
              "register_seeded_function"
            else
              "register_function"
          }("${ StringEscapeUtils.escapeString(name) }", (${ f.argTypes.map(dtype).mkString(",") }), ${ dtype(f.returnType) })""")
        }
    }
  }
}

abstract class RegistryFunctions {

  def registerAll(): Unit

  private val boxes = mutable.Map[String, Box[Type]]()

  def tv(name: String): TVariable =
    TVariable(name)

  def tv(name: String, cond: String): TVariable =
    TVariable(name, cond)

  def tnum(name: String): TVariable =
    tv(name, "numeric")

  def getRegion(mb: EmitMethodBuilder): Code[Region] = mb.getArg[Region](1)

  def wrapArg(mb: EmitMethodBuilder, t: Type): Code[_] => Code[_] = t match {
    case _: TBoolean => coerce[Boolean]
    case _: TInt32 => coerce[Int]
    case _: TInt64 => coerce[Long]
    case _: TFloat32 => coerce[Float]
    case _: TFloat64 => coerce[Double]
    case _: TCall => coerce[Int]
    case _: TString => c =>
      Code.invokeScalaObject[Region, Long, String](
        PString.getClass, "loadString",
        getRegion(mb), coerce[Long](c))
    case _ => c =>
      Code.invokeScalaObject[PType, Region, Long, Any](
        UnsafeRow.getClass, "read",
        mb.getPType(t.physicalType),
        getRegion(mb), coerce[Long](c))
  }

  def boxArg(mb: EmitMethodBuilder, t: Type): Code[_] => Code[Any] = t match {
    case _: TBoolean => c => Code.boxBoolean(coerce[Boolean](c))
    case _: TInt32 => c => Code.boxInt(coerce[Int](c))
    case _: TInt64 => c => Code.boxLong(coerce[Long](c))
    case _: TFloat32 => c => Code.boxFloat(coerce[Float](c))
    case _: TFloat64 => c => Code.boxDouble(coerce[Double](c))
    case _: TCall => c => Code.boxInt(coerce[Int](c))
    case _: TString => c =>
      Code.invokeScalaObject[Region, Long, String](
        PString.getClass, "loadString",
        getRegion(mb), coerce[Long](c))
    case _ => c =>
      Code.invokeScalaObject[PType, Region, Long, Any](
        UnsafeRow.getClass, "read",
        mb.getPType(t.physicalType),
        getRegion(mb), coerce[Long](c))
  }

  def unwrapReturn(mb: EmitMethodBuilder, t: Type): Code[_] => Code[_] = t match {
    case _: TBoolean => coerce[Boolean]
    case _: TInt32 => coerce[Int]
    case _: TInt64 => coerce[Long]
    case _: TFloat32 => coerce[Float]
    case _: TFloat64 => coerce[Double]
    case _: TString => c =>
      getRegion(mb).appendString(coerce[String](c))
    case _: TCall => coerce[Int]
    case TArray(_: TInt32, _) => c =>
      val srvb = new StagedRegionValueBuilder(mb, t.physicalType)
      val alocal = mb.newLocal[IndexedSeq[Int]]
      val len = mb.newLocal[Int]
      val v = mb.newLocal[java.lang.Integer]

      Code(
        alocal := coerce[IndexedSeq[Int]](c),
        len := alocal.invoke[Int]("size"),
        Code(
          srvb.start(len),
          Code.whileLoop(srvb.arrayIdx < len,
            v := Code.checkcast[java.lang.Integer](alocal.invoke[Int, java.lang.Object]("apply", srvb.arrayIdx)),
            v.isNull.mux(srvb.setMissing(), srvb.addInt(v.invoke[Int]("intValue"))),
            srvb.advance())),
        srvb.offset)
    case TArray(_: TFloat64, _) => c =>
      val srvb = new StagedRegionValueBuilder(mb, t.physicalType)
      val alocal = mb.newLocal[IndexedSeq[Double]]
      val len = mb.newLocal[Int]
      val v = mb.newLocal[java.lang.Double]

      Code(
        alocal := coerce[IndexedSeq[Double]](c),
        len := alocal.invoke[Int]("size"),
        Code(
          srvb.start(len),
          Code.whileLoop(srvb.arrayIdx < len,
            v := Code.checkcast[java.lang.Double](alocal.invoke[Int, java.lang.Object]("apply", srvb.arrayIdx)),
            v.isNull.mux(srvb.setMissing(), srvb.addDouble(v.invoke[Double]("doubleValue"))),
            srvb.advance())),
        srvb.offset)
    case TArray(_: TString, _) => c =>
      val srvb = new StagedRegionValueBuilder(mb, t.physicalType)
      val alocal = mb.newLocal[IndexedSeq[String]]
      val len = mb.newLocal[Int]
      val v = mb.newLocal[java.lang.String]

      Code(
        alocal := coerce[IndexedSeq[String]](c),
        len := alocal.invoke[Int]("size"),
        Code(
          srvb.start(len),
          Code.whileLoop(srvb.arrayIdx < len,
            v := Code.checkcast[java.lang.String](alocal.invoke[Int, java.lang.Object]("apply", srvb.arrayIdx)),
            v.isNull.mux(srvb.setMissing(), srvb.addString(v)),
            srvb.advance())),
        srvb.offset)
  }

  def registerCode(mname: String, aTypes: Array[Type], rType: Type)(impl: (EmitMethodBuilder, Array[Code[_]]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new IRFunctionWithoutMissingness {
      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def apply(mb: EmitMethodBuilder, args: Code[_]*): Code[_] = impl(mb, args.toArray)
    })
  }

  def registerCodeWithMissingness(mname: String, aTypes: Array[Type], rType: Type)(impl: (EmitMethodBuilder, Array[EmitTriplet]) => EmitTriplet) {
    IRFunctionRegistry.addIRFunction(new IRFunctionWithMissingness {
      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet = impl(mb, args.toArray)
    })
  }

  def registerScalaFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeScalaObject(cls, method, cts, args)(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerScalaFunction(mname: String, types: Type*)(cls: Class[_], method: String): Unit =
    registerScalaFunction(mname: String, types.init.toArray, types.last)(cls, method)

  def registerWrappedScalaFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    def ct(typ: Type): ClassTag[_] = typ match {
      case _: TString => classTag[String]
      case TArray(_: TInt32, _) => classTag[IndexedSeq[Int]]
      case TArray(_: TFloat64, _) => classTag[IndexedSeq[Double]]
      case TArray(_: TString, _) => classTag[IndexedSeq[String]]
      case TSet(_: TString, _) => classTag[Set[String]]
      case t => TypeToIRIntermediateClassTag(t)
    }

    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(ct(_).runtimeClass)
      val out = Code.invokeScalaObject(cls, method, cts, argTypes.zip(args).map { case (t, a) => wrapArg(mb, t)(a) })(ct(rType))
      unwrapReturn(mb, rType)(out)
    }
  }

  def registerWrappedScalaFunction(mname: String, a1: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(mname, Array(a1), rType)(cls, method)

  def registerWrappedScalaFunction(mname: String, a1: Type, a2: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(mname, Array(a1, a2), rType)(cls, method)

  def registerWrappedScalaFunction(mname: String, a1: Type, a2: Type, a3: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(mname, Array(a1, a2, a3), rType)(cls, method)

  def registerJavaStaticFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic(cls, method, cts, args)(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerJavaStaticFunction(mname: String, types: Type*)(cls: Class[_], method: String): Unit =
    registerJavaStaticFunction(mname, types.init.toArray, types.last)(cls, method)

  def registerIR(mname: String, argTypes: Array[Type], retType: Type)(f: Seq[IR] => IR) {
    IRFunctionRegistry.addIR(mname, argTypes, retType, f)
  }

  def registerCode(mname: String, rt: Type)(impl: EmitMethodBuilder => Code[_]): Unit =
    registerCode(mname, Array[Type](), rt) { (emb, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array()) => impl(mb)
      }
    }

  def registerCode[A1](mname: String, mt1: Type, rt: Type)(impl: (EmitMethodBuilder, Code[A1]) => Code[_]): Unit =
    registerCode(mname, Array(mt1), rt) {
      case (mb, Array(a1: Code[A1] @unchecked)) => impl(mb, a1)
    }

  def registerCode[A1, A2](mname: String, mt1: Type, mt2: Type, rt: Type)(impl: (EmitMethodBuilder, Code[A1], Code[A2]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2), rt) {
      case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked)) => impl(mb, a1, a2)
    }

  def registerCode[A1, A2, A3](mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type)
    (impl: (EmitMethodBuilder, Code[A1], Code[A2], Code[A3]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3), rt) {
      case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked, a3: Code[A3] @unchecked)) => impl(mb, a1, a2, a3)
    }

  def registerCode[A1, A2, A3, A4](mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type)
    (impl: (EmitMethodBuilder, Code[A1], Code[A2], Code[A3], Code[A4]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3, mt4), rt) {
      case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked, a3: Code[A3] @unchecked, a4: Code[A4] @unchecked)) => impl(mb, a1, a2, a3, a4)
    }

  def registerCode[A1, A2, A3, A4, A5](mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, rt: Type)
    (impl: (EmitMethodBuilder, Code[A1], Code[A2], Code[A3], Code[A4], Code[A5]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3, mt4, mt5), rt) {
      case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked, a3: Code[A3] @unchecked, a4: Code[A4] @unchecked, a5: Code[A5] @unchecked)) => impl(mb, a1, a2, a3, a4, a5)
    }

  def registerCodeWithMissingness(mname: String, rt: Type)(impl: EmitMethodBuilder => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array[Type](), rt) { case (mb, Array()) => impl(mb) }

  def registerCodeWithMissingness(mname: String, mt1: Type, rt: Type)(impl: (EmitMethodBuilder, EmitTriplet) => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array(mt1), rt) { case (mb, Array(a1)) => impl(mb, a1) }

  def registerCodeWithMissingness(mname: String, mt1: Type, mt2: Type, rt: Type)(impl: (EmitMethodBuilder, EmitTriplet, EmitTriplet) => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array(mt1, mt2), rt) { case (mb, Array(a1, a2)) => impl(mb, a1, a2) }

  def registerCodeWithMissingness(mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type)(impl: (EmitMethodBuilder, EmitTriplet, EmitTriplet, EmitTriplet, EmitTriplet) => EmitTriplet): Unit =
    registerCodeWithMissingness(mname, Array(mt1, mt2, mt3, mt4), rt) { case (mb, Array(a1, a2, a3, a4)) => impl(mb, a1, a2, a3, a4) }

  def registerIR(mname: String, retType: Type)(f: () => IR): Unit =
    registerIR(mname, Array[Type](), retType) { case Seq() => f() }

  def registerIR(mname: String, mt1: Type, retType: Type)(f: IR => IR): Unit =
    registerIR(mname, Array(mt1), retType) { case Seq(a1) => f(a1) }

  def registerIR(mname: String, mt1: Type, mt2: Type, retType: Type)(f: (IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2), retType) { case Seq(a1, a2) => f(a1, a2) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type, retType: Type)(f: (IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3), retType) { case Seq(a1, a2, a3) => f(a1, a2, a3) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, retType: Type)(f: (IR, IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3, mt4), retType) { case Seq(a1, a2, a3, a4) => f(a1, a2, a3, a4) }

  def registerSeeded(mname: String, aTypes: Array[Type], rType: Type)(impl: (EmitMethodBuilder, Long, Array[Code[_]]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new SeededIRFunction {
      val isDeterministic: Boolean = false

      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      def applySeeded(seed: Long, mb: EmitMethodBuilder, args: Code[_]*): Code[_] = impl(mb, seed, args.toArray)

      def applySeeded(seed: Long, mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet = {
        val setup = args.map(_.setup)
        val missing: Code[Boolean] = if (args.isEmpty) false else args.map(_.m).reduce(_ || _)
        val value = applySeeded(seed, mb, args.map(_.v): _*)

        EmitTriplet(setup, missing, value)
      }

      override val isStrict: Boolean = true
    })
  }

  def registerSeeded(mname: String, rType: Type)(impl: (EmitMethodBuilder, Long) => Code[_]): Unit =
    registerSeeded(mname, Array[Type](), rType) { (emb, seed, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array()) => impl(mb, seed)
      }
    }

  def registerSeeded[A1](mname: String, arg1: Type, rType: Type)(impl: (EmitMethodBuilder, Long, Code[A1]) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1), rType) { (emb, seed, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array(a1: Code[A1] @unchecked)) => impl(mb, seed, a1)
      }
    }

  def registerSeeded[A1, A2](mname: String, arg1: Type, arg2: Type, rType: Type)(impl: (EmitMethodBuilder, Long, Code[A1], Code[A2]) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1, arg2), rType) { (emb, seed, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked)) => impl(mb, seed, a1, a2)
      }
    }

  def registerSeeded[A1, A2, A3, A4](mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, rType: Type)(impl: (EmitMethodBuilder, Long, Code[A1], Code[A2], Code[A3], Code[A4]) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1, arg2, arg3, arg4), rType) { (emb, seed, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked, a3: Code[A3] @unchecked, a4: Code[A4] @unchecked)) => impl(mb, seed, a1, a2, a3, a4)
      }
    }
}

sealed abstract class IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet

  def getAsMethod(fb: EmitFunctionBuilder[_], args: Type*): EmitMethodBuilder = ???

  def returnType: Type

  override def toString: String = s"$name(${ argTypes.mkString(", ") }): $returnType"

  def unify(concrete: Seq[Type]): Boolean = {
    argTypes.length == concrete.length && {
      argTypes.foreach(_.clear())
      argTypes.zip(concrete).forall { case (i, j) => i.unify(j) }
    }
  }
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
    unify(args)
    val ts = argTypes.map(t => typeToTypeInfo(t.subst()))
    val methodbuilder = fb.newMethod((typeInfo[Region] +: ts).toArray, typeToTypeInfo(returnType.subst()))
    methodbuilder.emit(apply(methodbuilder, ts.zipWithIndex.map { case (a, i) => methodbuilder.getArg(i + 2)(a).load() }: _*))
    methodbuilder
  }

  def returnType: Type
}

abstract class IRFunctionWithMissingness extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet

  def returnType: Type
}

abstract class SeededIRFunction extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  private[this] var seed: Long = _
  def setSeed(s: Long): Unit = { seed = s }

  def applySeeded(seed: Long, mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet
  def apply(mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet =
    applySeeded(seed, mb, args: _*)

  def returnType: Type

  def isStrict: Boolean = false
}