package is.hail.expr.ir.functions

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types._
import is.hail.utils._
import is.hail.asm4s.coerce
import is.hail.experimental.ExperimentalFunctions
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.variant.Locus
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect._

object IRFunctionRegistry {
  private val userAddedFunctions: mutable.Set[(String, (Type, Seq[Type], Seq[Type]))] = mutable.HashSet.empty

  def clearUserFunctions() {
    userAddedFunctions.foreach { case (name, (rt, typeParameters, valueParameterTypes)) =>
      removeIRFunction(name, rt, typeParameters, valueParameterTypes) }
    userAddedFunctions.clear()
  }

  type IRFunctionSignature = (Seq[Type], Seq[Type], Type, Boolean)
  type IRFunctionImplementation = (Seq[Type], Seq[IR], Int) => IR

  val irRegistry: mutable.Map[String, mutable.Map[IRFunctionSignature, IRFunctionImplementation]] = new mutable.HashMap()

  val jvmRegistry: mutable.MultiMap[String, JVMFunction] =
    new mutable.HashMap[String, mutable.Set[JVMFunction]] with mutable.MultiMap[String, JVMFunction]

  private[this] def requireJavaIdentifier(name: String): Unit = {
    if (!isJavaIdentifier(name))
      throw new IllegalArgumentException(s"Illegal function name, not Java identifier: ${ name }")
  }

  def addJVMFunction(f: JVMFunction): Unit = {
    requireJavaIdentifier(f.name)
    jvmRegistry.addBinding(f.name, f)
  }

  def addIR(
    name: String,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type],
    returnType: Type,
    alwaysInline: Boolean,
    f: IRFunctionImplementation
  ): Unit = {
    requireJavaIdentifier(name)

    val m = irRegistry.getOrElseUpdate(name, new mutable.HashMap())
    m.update((typeParameters, valueParameterTypes, returnType, alwaysInline), f)
  }

  def pyRegisterIR(
    name: String,
    typeParamStrs: java.util.ArrayList[String],
    argNames: java.util.ArrayList[String],
    argTypeStrs: java.util.ArrayList[String],
    returnType: String,
    body: IR
  ): Unit = {
    requireJavaIdentifier(name)

    val typeParameters = typeParamStrs.asScala.map(IRParser.parseType).toFastIndexedSeq
    val valueParameterTypes = argTypeStrs.asScala.map(IRParser.parseType).toFastIndexedSeq
    userAddedFunctions += ((name, (body.typ, typeParameters, valueParameterTypes)))
    addIR(name,
      typeParameters,
      valueParameterTypes, IRParser.parseType(returnType), false, { (_, args, _) =>
        Subst(body,
          BindingEnv(Env[IR](argNames.asScala.zip(args): _*)))
      })
  }

  def removeIRFunction(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): Unit = {
    val m = irRegistry(name)
    m.remove((typeParameters, valueParameterTypes, returnType, false))
  }

  def lookupFunction(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): Option[JVMFunction] = {
    jvmRegistry.lift(name).map { fs => fs.filter(t => t.unify(typeParameters, valueParameterTypes, returnType)).toSeq }.getOrElse(FastSeq()) match {
      case Seq() => None
      case Seq(f) => Some(f)
      case _ => fatal(s"Multiple functions found that satisfy $name(${ valueParameterTypes.mkString(",") }).")
    }
  }

  def lookupFunctionOrFail(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): JVMFunction = {
    jvmRegistry.lift(name) match {
      case None =>
        fatal(s"no functions found with the name ${name}")
      case Some(functions) =>
        functions.filter(t => t.unify(typeParameters, valueParameterTypes, returnType)).toSeq match {
          case Seq() =>
            val prettyFunctionSignature = s"$name[${ typeParameters.mkString(", ") }](${ valueParameterTypes.mkString(", ") }): $returnType"
            val prettyMismatchedFunctionSignatures = functions.map(x => s"  $x").mkString("\n")
            fatal(
              s"No function found with the signature $prettyFunctionSignature.\n" +
              s"However, there are other functions with that name:\n$prettyMismatchedFunctionSignatures")
          case Seq(f) => f
          case _ => fatal(s"Multiple functions found that satisfy $name(${ valueParameterTypes.mkString(", ") }).")
        }
    }
  }

  def lookupIR(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): Option[(IRFunctionSignature, IRFunctionImplementation)] = {
    irRegistry.getOrElse(name, Map.empty).filter { case ((typeParametersFound: Seq[Type], valueParameterTypesFound: Seq[Type], _, _), _) =>
      typeParametersFound.length == typeParameters.length && {
        typeParametersFound.foreach(_.clear())
        (typeParametersFound, typeParameters).zipped.forall(_.unify(_))
      } && valueParameterTypesFound.length == valueParameterTypes.length && {
        valueParameterTypesFound.foreach(_.clear())
        (valueParameterTypesFound, valueParameterTypes).zipped.forall(_.unify(_))
      }
    }.toSeq match {
      case Seq() => None
      case Seq(kv) => Some(kv)
      case _ => fatal(s"Multiple functions found that satisfy $name(${valueParameterTypes.mkString(",")}).")
    }
  }

  def lookupSeeded(name: String, seed: Long, returnType: Type, arguments: Seq[Type]): Option[(Seq[IR]) => IR] = {
    lookupFunction(name, returnType, Array.empty[Type], arguments)
      .filter(_.isInstanceOf[SeededJVMFunction])
      .map { case f: SeededJVMFunction =>
        (irArguments: Seq[IR]) => ApplySeeded(name, irArguments, seed, f.returnType.subst())
      }
  }

  def lookupUnseeded(name: String, returnType: Type, arguments: Seq[Type]): Option[IRFunctionImplementation] =
    lookupUnseeded(name, returnType, Array.empty[Type], arguments)

  def lookupUnseeded(name: String, returnType: Type, typeParameters: Seq[Type], arguments: Seq[Type]): Option[IRFunctionImplementation] = {
    val validIR: Option[IRFunctionImplementation] = lookupIR(name, returnType, typeParameters, arguments).map {
      case ((_, _, _, inline), conversion) => (typeParametersPassed, args, errorID) =>
        val x = ApplyIR(name, typeParametersPassed, args, errorID)
        x.conversion = conversion
        x.inline = inline
        x
    }

    val validMethods = lookupFunction(name, returnType, typeParameters, arguments)
      .filter(!_.isInstanceOf[SeededJVMFunction]).map { f =>
        { (irValueParametersTypes: Seq[Type], irArguments: Seq[IR], errorID: Int) =>
          f match {
            case _: UnseededMissingnessObliviousJVMFunction =>
              Apply(name, irValueParametersTypes, irArguments, f.returnType.subst(), errorID)
            case _: UnseededMissingnessAwareJVMFunction =>
              ApplySpecial(name, irValueParametersTypes, irArguments, f.returnType.subst(), errorID)
          }
        }
      }

    (validIR, validMethods) match {
      case (None   , None)    => None
      case (None   , Some(x)) => Some(x)
      case (Some(x), None)    => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name(${ arguments.mkString(",") }).")
    }
  }

  Seq(
    ArrayFunctions,
    NDArrayFunctions,
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
    ExperimentalFunctions,
    ReferenceGenomeFunctions
  ).foreach(_.registerAll())

  def dumpFunctions(): Unit = {
    def dtype(t: Type): String = s"""dtype("${ StringEscapeUtils.escapeString(t.toString) }\")"""

    irRegistry.foreach { case (name, fns) =>
        fns.foreach { case ((typeParameters, valueParameterTypes, returnType, _), f) =>
          println(s"""register_function("${ StringEscapeUtils.escapeString(name) }", (${ typeParameters.map(dtype).mkString(",") }), (${ valueParameterTypes.map(dtype).mkString(",") }), ${ dtype(returnType) })""")
        }
    }

    jvmRegistry.foreach { case (name, fns) =>
        fns.foreach { f =>
          println(s"""${
            if (f.isInstanceOf[SeededJVMFunction])
              "register_seeded_function"
            else
              "register_function"
          }("${ StringEscapeUtils.escapeString(name) }", (${ f.typeParameters.map(dtype).mkString(",") }), (${ f.valueParameterTypes.map(dtype).mkString(",") }), ${ dtype(f.returnType) })""")
        }
    }
  }
}

object RegistryHelpers {
  def stupidUnwrapStruct(r: Region, value: Row, ptype: PType): Long = {
    assert(value != null)
    ptype.unstagedStoreJavaObject(value, r)
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

  def boxedTypeInfo(t: Type): TypeInfo[_ >: Null] = t match {
    case TBoolean => classInfo[java.lang.Boolean]
    case TInt32 => classInfo[java.lang.Integer]
    case TInt64 => classInfo[java.lang.Long]
    case TFloat32 => classInfo[java.lang.Float]
    case TFloat64 => classInfo[java.lang.Double]
    case TCall => classInfo[java.lang.Integer]
    case TString => classInfo[java.lang.String]
    case _: TLocus => classInfo[Locus]
    case _ => classInfo[AnyRef]
  }

  def scodeToJavaValue(cb: EmitCodeBuilder, r: Value[Region], sc: SCode): Code[AnyRef] = {
    sc.st match {
      case SInt32 => Code.boxInt(sc.asInt32.intCode(cb))
      case SInt64 => Code.boxLong(sc.asInt64.longCode(cb))
      case SFloat32 => Code.boxFloat(sc.asFloat32.floatCode(cb))
      case SFloat64 => Code.boxDouble(sc.asFloat64.doubleCode(cb))
      case SBoolean => Code.boxBoolean(sc.asBoolean.boolCode(cb))
      case _: SCall => Code.boxInt(sc.asCall.loadCanonicalRepresentation(cb))
      case _: SString => sc.asString.loadString()
      case _: SLocus => sc.asLocus.getLocusObj(cb)
      case t =>
        val pt = t.canonicalPType()
        val addr = pt.store(cb, r, sc, deepCopy = false)
        Code.invokeScalaObject3[PType, Region, Long, AnyRef](
          UnsafeRow.getClass, "readAnyRef",
          cb.emb.getPType(pt),
          r, addr)

    }
  }

  def unwrapReturn(cb: EmitCodeBuilder, r: Value[Region], st: SType, value: Code[_]): SCode = st.virtualType match {
    case TBoolean => primitive(coerce[Boolean](value))
    case TInt32 => primitive(coerce[Int](value))
    case TInt64 => primitive(coerce[Long](value))
    case TFloat32 => primitive(coerce[Float](value))
    case TFloat64 => primitive(coerce[Double](value))
    case TString =>
      val sst = st.asInstanceOf[SJavaString.type]
      sst.constructFromString(cb, r, coerce[String](value))
    case TCall =>
      assert(st == SCanonicalCall)
      new SCanonicalCallCode(coerce[Int](value))
    case TArray(TInt32) =>
      val ast = st.asInstanceOf[SIndexablePointer]
      val pca = ast.pType.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Int]]("unrwrap_return_array_int32_arr", coerce[IndexedSeq[Int]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_int32_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Integer]("unwrap_return_array_int32_elt",
          Code.checkcast[java.lang.Integer](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(elt.invoke[Int]("intValue")))
      }
    case TArray(TFloat64) =>
      val ast = st.asInstanceOf[SIndexablePointer]
      val pca = ast.pType.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Double]]("unrwrap_return_array_float64_arr", coerce[IndexedSeq[Double]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_float64_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Double]("unwrap_return_array_float64_elt",
          Code.checkcast[java.lang.Double](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(elt.invoke[Double]("doubleValue")))
      }
    case TArray(TString) =>
      val ast = st.asInstanceOf[SJavaArrayString]
      ast.construct(coerce[Array[String]](value))
    case t: TBaseStruct =>
      val sst = st.asInstanceOf[SBaseStructPointer]
      val pt = sst.pType.asInstanceOf[PCanonicalBaseStruct]
      val addr = Code.invokeScalaObject3[Region, Row, PType, Long](
        RegistryHelpers.getClass, "stupidUnwrapStruct", r.region, coerce[Row](value), cb.emb.ecb.getPType(pt))
      new SBaseStructPointerCode(SBaseStructPointer(pt.setRequired(false).asInstanceOf[PBaseStruct]), addr)
  }

  def registerSCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[SType]) => SType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, Array[SCode], Value[Int]) => SCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessObliviousJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnType) {
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnSType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: SCode*): SCode =
          impl(r, cb, typeParameters, returnSType, args.toArray, errorID)
      })
  }

  def registerCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[SType]) => SType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion, EmitCodeBuilder, SType, Array[Type], Array[SCode]) => Code[_]
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessObliviousJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnType) {
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnSType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: SCode*): SCode = {
          assert(unify(typeParameters, args.map(_.st.virtualType), returnSType.virtualType))
          returnSType.fromCodes(FastIndexedSeq(impl(r, cb, returnSType, typeParameters.toArray, args.toArray)))
        }
      })
  }

  def registerEmitCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[EmitType]) => EmitType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion, SType,  Value[Int], Array[EmitCode]) => EmitCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessAwareJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnType) {
        override def apply(r: EmitRegion, rpt: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): EmitCode = {
          assert(unify(typeParameters, args.map(_.st.virtualType), rpt.virtualType))
          impl(r, rpt, errorID, args.toArray)
        }
      })
  }

  def registerIEmitCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[EmitType]) => EmitType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitCodeBuilder, Value[Region], SType , Value[Int], Array[EmitCode]) => IEmitCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessAwareJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnType) {
        override def apply(
          cb: EmitCodeBuilder,
          r: Value[Region],
          rpt: SType,
          typeParameters: Seq[Type],
          errorID: Value[Int],
          args: EmitCode*
        ): IEmitCode = {
          val res = impl(cb, r, rpt, errorID, args.toArray)
          if (res.emitType != calculateReturnType(rpt.virtualType, args.map(_.emitType)))
            throw new RuntimeException(s"type mismatch while registering $name" +
              s"\n  got ${ res.emitType }, got ${ calculateReturnType(rpt.virtualType, args.map(_.emitType)) }")
          res
        }
        override def apply(r: EmitRegion, rpt: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): EmitCode = {
          EmitCode.fromI(r.mb) { cb =>
            apply(cb, r.region, rpt, typeParameters, errorID, args: _*)
          }
        }
      })
  }

  def registerScalaFunction(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[SType]) => SType
  )(
    cls: Class[_],
    method: String
  ) {
    registerSCode(name, valueParameterTypes, returnType, calculateReturnType) { case (r, cb, _, rt, args, _) =>
      val cts = valueParameterTypes.map(PrimitiveTypeToIRIntermediateClassTag(_).runtimeClass)
      rt.fromCodes(FastIndexedSeq(
        Code.invokeScalaObject(cls, method, cts, args.map { a => SType.extractPrimCode(cb, a) })(PrimitiveTypeToIRIntermediateClassTag(returnType))
      ))
    }
  }

  def registerWrappedScalaFunction(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[SType]) => SType
  )(
    cls: Class[_],
    method: String
  ) {
    def ct(typ: Type): ClassTag[_] = typ match {
      case TString => classTag[String]
      case TArray(TInt32) => classTag[IndexedSeq[Int]]
      case TArray(TFloat64) => classTag[IndexedSeq[Double]]
      case TArray(TString) => classTag[Array[String]]
      case TSet(TString) => classTag[Set[String]]
      case TDict(TString, TString) => classTag[Map[String, String]]
      case TCall => classTag[Int]
      case t => PrimitiveTypeToIRIntermediateClassTag(t)
    }

    def wrap(cb: EmitCodeBuilder, r: Value[Region], code: SCode): Code[_] = code.st.virtualType match {
      case t if t.isPrimitive => SType.extractPrimCode(cb, code)
      case TCall => code.asCall.loadCanonicalRepresentation(cb)
      case TArray(TString) => code.st match {
        case _: SJavaArrayString => code.asInstanceOf[SJavaArrayStringCode].array
        case _ =>
          val sv = code.asIndexable.memoize(cb, "scode_array_string")
          val arr = cb.newLocal[Array[String]]("scode_array_string", Code.newArray[String](sv.loadLength()))
          sv.foreach(cb) { case (cb, idx, elt) =>
            elt.consume(cb,
              (),
              { sc =>
                cb += (arr(idx) = sc.asString.loadString())
              })
          }
          arr
      }
      case _ => scodeToJavaValue(cb, r, code)
    }

    registerSCode(name, valueParameterTypes, returnType, calculateReturnType) { case (r, cb, _, rt, args, _) =>
      val cts = valueParameterTypes.map(ct(_).runtimeClass)
      try {
        unwrapReturn(cb, r.region, rt,
          Code.invokeScalaObject(cls, method, cts, args.map { a => wrap(cb, r.region, a) })(ct(returnType)))
      } catch {
        case e: Throwable => throw new RuntimeException(s"error while registering function $name", e)
      }
    }
  }

  def registerWrappedScalaFunction1(name: String, a1: Type, returnType: Type,  pt: (Type, SType) => SType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1), returnType, unwrappedApply(pt))(cls, method)

  def registerWrappedScalaFunction2(name: String, a1: Type, a2: Type, returnType: Type, pt: (Type, SType, SType) => SType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1, a2), returnType, unwrappedApply(pt))(cls, method)

  def registerWrappedScalaFunction3(name: String, a1: Type, a2: Type, a3: Type, returnType: Type,
    pt: (Type, SType, SType, SType) => SType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1, a2, a3), returnType, unwrappedApply(pt))(cls, method)

  def registerJavaStaticFunction(name: String, valueParameterTypes: Array[Type], returnType: Type, pt: (Type, Seq[SType]) => SType)(cls: Class[_], method: String) {
    registerCode(name, valueParameterTypes, returnType, pt) { case (r, cb, rt, _, args) =>
      val cts = valueParameterTypes.map(PrimitiveTypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic(cls, method, cts, args.map(a => SType.extractPrimCode(cb, a)))(PrimitiveTypeToIRIntermediateClassTag(returnType))
    }
  }

  def registerIR(name: String, valueParameterTypes: Array[Type], returnType: Type, inline: Boolean = false, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], Seq[IR], Int) => IR): Unit =
    IRFunctionRegistry.addIR(name, typeParameters, valueParameterTypes, returnType, inline, f)

  def registerSCode1(name: String, mt1: Type, rt: Type, pt: (Type, SType) => SType)(impl: (EmitRegion, EmitCodeBuilder, SType, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1), errorID) => impl(r, cb, rt, a1, errorID)
    }

  def registerSCode1t(name: String, typeParams: Array[Type], mt1: Type, rt: Type, pt: (Type, SType) => SType)(impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1), rt, unwrappedApply(pt), typeParameters = typeParams) {
      case (r, cb, typeParams, rt, Array(a1), errorID) => impl(r, cb, typeParams, rt, a1, errorID)
    }

  def registerSCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SCode, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2) , errorID) => impl(r, cb, rt, a1, a2, errorID)
    }

  def registerSCode2t(name: String, typeParams: Array[Type], mt1: Type, mt2: Type, rt: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, SCode, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1, mt2), rt, unwrappedApply(pt), typeParameters = typeParams) {
      case (r, cb, typeParams, rt, Array(a1, a2), errorID) => impl(r, cb, typeParams, rt, a1, a2, errorID)
    }

  def registerSCode3(name: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, pt: (Type, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SCode, SCode, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1, mt2, mt3), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3), errorID) => impl(r, cb, rt, a1, a2, a3, errorID)
    }

  def registerSCode4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (Type, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SCode, SCode, SCode, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4), errorID) => impl(r, cb, rt, a1, a2, a3, a4, errorID)
    }

  def registerSCode4t(name: String, typeParams: Array[Type], mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type,
    pt: (Type, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, SCode, SCode, SCode, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt), typeParams) {
      case (r, cb, typeParams, rt, Array(a1, a2, a3, a4), errorID) => impl(r, cb, typeParams, rt, a1, a2, a3, a4, errorID)
    }


  def registerSCode5(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, rt: Type, pt: (Type, SType, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SCode, SCode, SCode, SCode, SCode, Value[Int]) => SCode): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4, mt5), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4, a5), errorID) => impl(r, cb, rt, a1, a2, a3, a4, a5, errorID)
    }

  def registerCode1(name: String, mt1: Type, rt: Type, pt: (Type, SType) => SType)(impl: (EmitCodeBuilder, EmitRegion, SType, SCode) => Code[_]): Unit =
    registerCode(name, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, cb, rt, _, Array(a1)) => impl(cb, r, rt, a1)
    }

  def registerCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitCodeBuilder, EmitRegion, SType, SCode, SCode) => Code[_]): Unit =
    registerCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) {
      case (r, cb, rt, _, Array(a1, a2)) => impl(cb, r, rt, a1, a2)
    }

  def registerIEmitCode1(name: String, mt1: Type, rt: Type, pt: (Type, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1)) =>
      impl(cb, r, rt,  errorID, a1)
    }

  def registerIEmitCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, EmitType, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1, a2)) =>
      impl(cb, r, rt, errorID, a1, a2)
    }

  def registerIEmitCode4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (Type, EmitType, EmitType, EmitType, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode, EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1, a2, a3, a4)) =>
      impl(cb, r, rt, errorID, a1, a2, a3, a4)
    }

  def registerIEmitCode6(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, mt6: Type, rt: Type, pt: (Type, EmitType, EmitType, EmitType, EmitType, EmitType, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode, EmitCode, EmitCode, EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3, mt4, mt5, mt6), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1, a2, a3, a4, a5, a6)) =>
      impl(cb, r, rt, errorID, a1, a2, a3, a4, a5, a6)
    }

  def registerEmitCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, EmitType, EmitType) => EmitType)
    (impl: (EmitRegion, SType, Value[Int], EmitCode, EmitCode) => EmitCode): Unit =
    registerEmitCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) { case (r, rt, errorID, Array(a1, a2)) => impl(r, rt, errorID, a1, a2) }

  def registerIR1(name: String, mt1: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, Int) => IR): Unit =
    registerIR(name, Array(mt1), returnType, typeParameters = typeParameters) { case (t, Seq(a1), errorID) => f(t, a1, errorID) }

  def registerIR2(name: String, mt1: Type, mt2: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, IR, Int) => IR): Unit =
    registerIR(name, Array(mt1, mt2), returnType, typeParameters = typeParameters) { case (t, Seq(a1, a2), errorID) => f(t, a1, a2, errorID) }

  def registerIR3(name: String, mt1: Type, mt2: Type, mt3: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, IR, IR, Int) => IR): Unit =
    registerIR(name, Array(mt1, mt2, mt3), returnType, typeParameters = typeParameters) { case (t, Seq(a1, a2, a3), errorID) => f(t, a1, a2, a3, errorID) }

  def registerIR4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, IR, IR, IR, Int) => IR): Unit =
    registerIR(name, Array(mt1, mt2, mt3, mt4), returnType, typeParameters = typeParameters) { case (t, Seq(a1, a2, a3, a4), errorID) => f(t, a1, a2, a3, a4, errorID) }

  def registerSeeded(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    computeReturnType: (Type, Seq[SType]) => SType
  )(
    impl: (EmitCodeBuilder, Value[Region], SType, Long, Array[SCode]) => SCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new SeededMissingnessObliviousJVMFunction(name, valueParameterTypes, returnType, computeReturnType) {
        val isDeterministic: Boolean = false

        def applySeeded(cb: EmitCodeBuilder, seed: Long, r: Value[Region], rpt: SType, args: SCode*): SCode = {
          assert(unify(Array.empty[Type], args.map(_.st.virtualType), rpt.virtualType))
          impl(cb, r, rpt, seed, args.toArray)
        }

        def applySeededI(seed: Long, cb: EmitCodeBuilder, r: Value[Region], rpt: SType, args: EmitCode*): IEmitCode = {
          IEmitCode.multiMapEmitCodes(cb, args.toFastIndexedSeq) {
            argPCs => applySeeded(cb, seed, r, rpt, argPCs: _*)
          }
        }

        override val isStrict: Boolean = true
      })
  }

  def registerSeeded0(name: String, returnType: Type, pt: SType)(impl: (EmitCodeBuilder, Value[Region], SType, Long) => SCode): Unit =
    registerSeeded(name, Array[Type](), returnType, if (pt == null) null else (_: Type, _: Seq[SType]) => pt) { case (cb, r, rt, seed, _) => impl(cb, r, rt, seed) }

  def registerSeeded1(name: String, arg1: Type, returnType: Type, pt: (Type, SType) => SType)(impl: (EmitCodeBuilder, Value[Region], SType, Long, SCode) => SCode): Unit =
    registerSeeded(name, Array(arg1), returnType, unwrappedApply(pt)) {
      case (cb, r, rt, seed, Array(a1)) => impl(cb, r, rt, seed, a1)
    }

  def registerSeeded2(name: String, arg1: Type, arg2: Type, returnType: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Long, SCode, SCode) => SCode): Unit =
    registerSeeded(name, Array(arg1, arg2), returnType, unwrappedApply(pt)) { case
      (cb, r, rt, seed, Array(a1, a2)) =>
      impl(cb, r, rt, seed, a1, a2)
    }

  def registerSeeded4(name: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, returnType: Type, pt: (Type, SType, SType, SType, SType) => SType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Long, SCode, SCode, SCode, SCode) => SCode): Unit =
    registerSeeded(name, Array(arg1, arg2, arg3, arg4), returnType, unwrappedApply(pt)) {
      case (cb, r, rt, seed, Array(a1, a2, a3, a4)) => impl(cb, r, rt, seed, a1, a2, a3, a4)
    }
}

sealed abstract class JVMFunction {
  def name: String

  def typeParameters: Seq[Type]

  def valueParameterTypes: Seq[Type]

  def returnType: Type

  def computeReturnEmitType(returnType: Type, valueParameterTypes: Seq[EmitType]): EmitType

  def apply(mb: EmitRegion, returnType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): EmitCode

  override def toString: String = s"$name[${ typeParameters.mkString(", ") }](${ valueParameterTypes.mkString(", ") }): $returnType"

  def unify(typeArguments: Seq[Type], valueArgumentTypes: Seq[Type], returnTypeIn: Type): Boolean = {
    val concrete = (typeArguments ++ valueArgumentTypes) :+ returnTypeIn
    val types = (typeParameters ++ valueParameterTypes) :+ returnType
    types.length == concrete.length && {
      types.foreach(_.clear())
      types.zip(concrete).forall { case (i, j) => i.unify(j) }
    }
  }
}

object MissingnessObliviousJVMFunction {
  def returnSType(computeStrictReturnEmitType: (Type, Seq[SType]) => SType)(returnType: Type, valueParameterTypes: Seq[SType]): SType = {
    if (computeStrictReturnEmitType == null)
      SType.canonical(returnType)
    else
      computeStrictReturnEmitType(returnType, valueParameterTypes)
  }
}

abstract class UnseededMissingnessObliviousJVMFunction (
  override val name: String,
  override val typeParameters: Seq[Type],
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessObliviousComputeReturnType: (Type, Seq[SType]) => SType
) extends JVMFunction {
  override def computeReturnEmitType(returnType: Type, valueParameterTypes: Seq[EmitType]): EmitType = {
    EmitType(computeStrictReturnEmitType(returnType, valueParameterTypes.map(_.st)), valueParameterTypes.forall(_.required))
  }
  def computeStrictReturnEmitType(returnType: Type, valueParameterTypes: Seq[SType]): SType =
    MissingnessObliviousJVMFunction.returnSType(missingnessObliviousComputeReturnType)(returnType, valueParameterTypes)

  def apply(r: EmitRegion, cb: EmitCodeBuilder, returnSType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: SCode*): SCode

  def apply(r: EmitRegion, returnType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): EmitCode = {
    EmitCode.fromI(r.mb)(cb => IEmitCode.multiMapEmitCodes(cb, args.toFastIndexedSeq) { args =>
      apply(r, cb, returnType, typeParameters, errorID, args: _*)
    })
  }

  def getAsMethod[C](cb: EmitClassBuilder[C], rpt: SType, typeParameters: Seq[Type], args: SType*): EmitMethodBuilder[C] = {
    val unified = unify(typeParameters, args.map(_.virtualType), rpt.virtualType)
    assert(unified, name)
    val methodbuilder = cb.genEmitMethod(name, FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[Int]) ++ args.map(_.paramType), rpt.paramType)
    methodbuilder.emitSCode(cb => apply(EmitRegion.default(methodbuilder),
      cb,
      rpt,
      typeParameters,
      methodbuilder.getCodeParam[Int](2),
      (0 until args.length).map(i => methodbuilder.getSCodeParam(i + 3)): _*))
    methodbuilder
  }
}

object MissingnessAwareJVMFunction {
  def returnSType(calculateReturnType: (Type, Seq[EmitType]) => EmitType)(returnType: Type, valueParameterTypes: Seq[EmitType]): EmitType =
    if (calculateReturnType == null) EmitType(SType.canonical(returnType), false)
    else calculateReturnType(returnType, valueParameterTypes)
}

abstract class UnseededMissingnessAwareJVMFunction (
  override val name: String,
  override val typeParameters: Seq[Type],
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessAwareComputeReturnSType: (Type, Seq[EmitType]) => EmitType
) extends JVMFunction {
  override def computeReturnEmitType(returnType: Type, valueParameterTypes: Seq[EmitType]): EmitType =
    MissingnessAwareJVMFunction.returnSType(missingnessAwareComputeReturnSType)(returnType, valueParameterTypes)

  def apply(cb: EmitCodeBuilder,
    r: Value[Region],
    rpt: SType,
    typeParameters: Seq[Type],
    errorID: Value[Int],
    args: EmitCode*
  ): IEmitCode = {
    ???
  }
}

abstract class SeededJVMFunction (
  override val name: String,
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type
) extends JVMFunction {
  def typeParameters: Seq[Type] = Seq.empty[Type]

  private[this] var seed: Long = _

  def setSeed(s: Long): Unit = { seed = s }

  def applySeededI(seed: Long, cb: EmitCodeBuilder, region: Value[Region], rpt: SType, args: EmitCode*): IEmitCode

  def apply(region: EmitRegion, rpt: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): EmitCode =
    fatal("seeded functions must go through IEmitCode path")

  def apply(region: EmitRegion, rpt: SType, args: EmitCode*): EmitCode =
    fatal("seeded functions must go through IEmitCode path")

  def isStrict: Boolean = false
}

abstract class SeededMissingnessObliviousJVMFunction (
  override val name: String,
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessObliviousreturnSType: (Type, Seq[SType]) => SType
) extends SeededJVMFunction(name, valueParameterTypes, returnType) {
  override def computeReturnEmitType(returnType: Type, valueParameterTypes: Seq[EmitType]): EmitType = {
    EmitType(computeStrictReturnEmitType(returnType, valueParameterTypes.map(_.st)), valueParameterTypes.forall(_.required))
  }

  def computeStrictReturnEmitType(returnType: Type, valueParameterTypes: Seq[SType]): SType =
    MissingnessObliviousJVMFunction.returnSType(missingnessObliviousreturnSType)(returnType, valueParameterTypes)
}

abstract class SeededMissingnessAwareJVMFunction (
  override val name: String,
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessAwarereturnSType: (Type, Seq[EmitType]) => EmitType
) extends SeededJVMFunction(name, valueParameterTypes, returnType) {
  override def computeReturnEmitType(returnType: Type, valueParameterTypes: Seq[EmitType]): EmitType =
    MissingnessAwareJVMFunction.returnSType(missingnessAwarereturnSType)(returnType, valueParameterTypes)
}
