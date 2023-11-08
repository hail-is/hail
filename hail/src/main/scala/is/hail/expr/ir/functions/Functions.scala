package is.hail.expr.ir.functions

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.experimental.ExperimentalFunctions
import is.hail.expr.ir._
import is.hail.io.bgen.BGENFunctions
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.{Locus, ReferenceGenome}
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

    val typeParameters = typeParamStrs.asScala.map(IRParser.parseType).toFastSeq
    val valueParameterTypes = argTypeStrs.asScala.map(IRParser.parseType).toFastSeq
    userAddedFunctions += ((name, (body.typ, typeParameters, valueParameterTypes)))
    addIR(name,
      typeParameters,
      valueParameterTypes, IRParser.parseType(returnType), false, { (_, args, _) =>
        Subst(body,
          BindingEnv(Env[IR](argNames.asScala.zip(args): _*)))
      })
  }

  def pyRegisterIRForServiceBackend(
    ctx: ExecuteContext,
    name: String,
    typeParamStrs: Array[String],
    argNames: Array[String],
    argTypeStrs: Array[String],
    returnType: String,
    bodyStr: String
  ): Unit = {
    requireJavaIdentifier(name)

    val typeParameters = typeParamStrs.map(IRParser.parseType).toFastSeq
    val valueParameterTypes = argTypeStrs.map(IRParser.parseType).toFastSeq
    val refMap = BindingEnv.eval(argNames.zip(valueParameterTypes): _*)
    val body = IRParser.parse_value_ir(
      bodyStr,
      IRParserEnvironment(ctx, Map()),
      refMap)

    userAddedFunctions += ((name, (body.typ, typeParameters, valueParameterTypes)))
    addIR(
      name,
      typeParameters,
      valueParameterTypes,
      IRParser.parseType(returnType),
      false,
      { (_, args, _) =>
        Subst(body,
          BindingEnv(Env[IR](argNames.zip(args): _*)))
      }
    )
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
        fatal(s"no functions found with the signature $name(${valueParameterTypes.mkString(", ")}): $returnType")
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

  def lookupSeeded(name: String, staticUID: Long, returnType: Type, arguments: Seq[Type]): Option[(Seq[IR], IR) => IR] = {
    lookupFunction(name, returnType, Array.empty[Type], TRNGState +: arguments)
      .map { f =>
        (irArguments: Seq[IR], rngState: IR) => ApplySeeded(name, irArguments, rngState, staticUID, f.returnType.subst())
      }
  }

  def lookupUnseeded(name: String, returnType: Type, arguments: Seq[Type]): Option[IRFunctionImplementation] =
    lookupUnseeded(name, returnType, Array.empty[Type], arguments)

  def lookupUnseeded(name: String, returnType: Type, typeParameters: Seq[Type], arguments: Seq[Type]): Option[IRFunctionImplementation] = {
    val validIR: Option[IRFunctionImplementation] = lookupIR(name, returnType, typeParameters, arguments).map {
      case ((_, _, _, inline), conversion) => (typeParametersPassed, args, errorID) =>
        val x = ApplyIR(name, returnType, typeParametersPassed, args, errorID)
        x.conversion = conversion
        x.inline = inline
        x
    }

    val validMethods = lookupFunction(name, returnType, typeParameters, arguments)
      .map { f =>
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
    ReferenceGenomeFunctions,
    BGENFunctions,
    ApproxCDFFunctions
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
          println(s"""register_function("${ StringEscapeUtils.escapeString(name) }", (${ f.typeParameters.map(dtype).mkString(",") }), (${ f.valueParameterTypes.map(dtype).mkString(",") }), ${ dtype(f.returnType) })""")
        }
    }
  }
}

object RegistryHelpers {
  def stupidUnwrapStruct(rgs: Map[String, ReferenceGenome], r: Region, value: Row, ptype: PType): Long = {
    assert(value != null)
    ptype.unstagedStoreJavaObject(HailStateManager(rgs), value, r)
  }

  def stupidUnwrapArray(rgs: Map[String, ReferenceGenome], r: Region, value: IndexedSeq[Annotation], ptype: PType): Long = {
    assert(value != null)
    ptype.unstagedStoreJavaObject(HailStateManager(rgs), value, r)
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

  def svalueToJavaValue(cb: EmitCodeBuilder, r: Value[Region], sc: SValue, safe: Boolean = false): Value[AnyRef] = {
    sc.st match {
      case SInt32 => cb.memoize(Code.boxInt(sc.asInt32.value))
      case SInt64 => cb.memoize(Code.boxLong(sc.asInt64.value))
      case SFloat32 => cb.memoize(Code.boxFloat(sc.asFloat32.value))
      case SFloat64 => cb.memoize(Code.boxDouble(sc.asFloat64.value))
      case SBoolean => cb.memoize(Code.boxBoolean(sc.asBoolean.value))
      case _: SCall => cb.memoize(Code.boxInt(sc.asCall.canonicalCall(cb)))
      case _: SString => sc.asString.loadString(cb)
      case _: SLocus => sc.asLocus.getLocusObj(cb)
      case t =>
        val pt = PType.canonical(t.storageType())
        val addr = pt.store(cb, r, sc, deepCopy = false)
        cb.memoize(Code.invokeScalaObject3[PType, Region, Long, AnyRef](
          if (safe) SafeRow.getClass else UnsafeRow.getClass, "readAnyRef",
          cb.emb.getPType(pt),
          r, addr))

    }
  }

  def unwrapReturn(cb: EmitCodeBuilder, r: Value[Region], st: SType, value: Code[_]): SValue = st.virtualType match {
    case TBoolean => primitive(cb.memoize(coerce[Boolean](value)))
    case TInt32 => primitive(cb.memoize(coerce[Int](value)))
    case TInt64 => primitive(cb.memoize(coerce[Long](value)))
    case TFloat32 => primitive(cb.memoize(coerce[Float](value)))
    case TFloat64 => primitive(cb.memoize(coerce[Double](value)))
    case TString =>
      val sst = st.asInstanceOf[SJavaString.type]
      sst.constructFromString(cb, r, coerce[String](value))
    case TCall =>
      assert(st == SCanonicalCall)
      new SCanonicalCallValue(cb.memoize(coerce[Int](value)))
    case TArray(TInt32) =>
      val ast = st.asInstanceOf[SIndexablePointer]
      val pca = ast.pType.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Int]]("unrwrap_return_array_int32_arr", coerce[IndexedSeq[Int]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_int32_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Integer]("unwrap_return_array_int32_elt",
          Code.checkcast[java.lang.Integer](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(cb.memoize(elt.invoke[Int]("intValue"))))
      }
    case TArray(TInt64) =>
      val ast = st.asInstanceOf[SIndexablePointer]
      val pca = ast.pType.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Int]]("unrwrap_return_array_int64_arr", coerce[IndexedSeq[Int]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_int64_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Long]("unwrap_return_array_int64_elt",
          Code.checkcast[java.lang.Long](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(cb.memoize(elt.invoke[Long]("longValue"))))
      }
    case TArray(TFloat64) =>
      val ast = st.asInstanceOf[SIndexablePointer]
      val pca = ast.pType.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Double]]("unrwrap_return_array_float64_arr", coerce[IndexedSeq[Double]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_float64_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Double]("unwrap_return_array_float64_elt",
          Code.checkcast[java.lang.Double](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(cb.memoize(elt.invoke[Double]("doubleValue"))))
      }
    case TArray(TString) =>
      val ast = st.asInstanceOf[SJavaArrayString]
      ast.construct(cb, coerce[Array[String]](value))
    case t: TBaseStruct =>
      val sst = st.asInstanceOf[SBaseStructPointer]
      val pt = sst.pType.asInstanceOf[PCanonicalBaseStruct]
      val addr = cb.memoize(Code.invokeScalaObject4[Map[String, ReferenceGenome], Region, Row, PType, Long](
        RegistryHelpers.getClass, "stupidUnwrapStruct", cb.emb.ecb.emodb.referenceGenomeMap, r.region, coerce[Row](value), cb.emb.ecb.getPType(pt)))
      new SBaseStructPointerValue(SBaseStructPointer(pt.setRequired(false).asInstanceOf[PBaseStruct]), addr)
    case TArray(t: TBaseStruct) =>
      val ast = st.asInstanceOf[SIndexablePointer]
      val pca = ast.pType.asInstanceOf[PCanonicalArray]
      val array = cb.memoize(Code.invokeScalaObject4[Map[String, ReferenceGenome], Region, IndexedSeq[Annotation], PType, Long](
        RegistryHelpers.getClass, "stupidUnwrapArray", cb.emb.ecb.emodb.referenceGenomeMap, r.region, coerce[IndexedSeq[Annotation]](value), cb.emb.ecb.getPType(pca)))
      new SIndexablePointerValue(ast, array, cb.memoize(pca.loadLength(array)), cb.memoize(pca.firstElementOffset(array)))
  }

  def registerSCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnType: (Type, Seq[SType]) => SType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, Array[SValue], Value[Int]) => SValue
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessObliviousJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnType) {
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnSType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: SValue*): SValue =
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
    impl: (EmitRegion, EmitCodeBuilder, SType, Array[Type], Array[SValue]) => Value[_]
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessObliviousJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnType) {
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnSType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: SValue*): SValue = {
          assert(unify(typeParameters, args.map(_.st.virtualType), returnSType.virtualType))
          val returnValue = impl(r, cb, returnSType, typeParameters.toArray, args.toArray)
          returnSType.fromValues(FastSeq(returnValue))
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

      val returnValue = cb.memoizeAny(
        Code.invokeScalaObject(cls, method, cts, args.map { a => SType.extractPrimValue(cb, a).get })(PrimitiveTypeToIRIntermediateClassTag(returnType)),
        rt.settableTupleTypes()(0))
      rt.fromValues(FastSeq(returnValue))
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

    def wrap(cb: EmitCodeBuilder, r: Value[Region], code: SValue): Value[_] = code.st.virtualType match {
      case t if t.isPrimitive => SType.extractPrimValue(cb, code)
      case TCall => code.asCall.canonicalCall(cb)
      case TArray(TString) => code.st match {
        case _: SJavaArrayString => cb.memoize(code.asInstanceOf[SJavaArrayStringValue].array)
        case _ =>
          val sv = code.asIndexable
          val arr = cb.newLocal[Array[String]]("scode_array_string", Code.newArray[String](sv.loadLength()))
          sv.forEachDefined(cb) { case (cb, idx, elt) =>
            cb += (arr(idx) = elt.asString.loadString(cb))
          }
          arr
      }
      case _ => svalueToJavaValue(cb, r, code)
    }

    registerSCode(name, valueParameterTypes, returnType, calculateReturnType) { case (r, cb, _, rt, args, _) =>
      val cts = valueParameterTypes.map(ct(_).runtimeClass)
      try {
        unwrapReturn(cb, r.region, rt,
          Code.invokeScalaObject(cls, method, cts, args.map { a => wrap(cb, r.region, a).get })(ct(returnType)))
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

  def registerWrappedScalaFunction4(name: String, a1: Type, a2: Type, a3: Type, a4: Type, returnType: Type,
                                    pt: (Type, SType, SType, SType, SType) => SType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1, a2, a3, a4), returnType, unwrappedApply(pt))(cls, method)

  def registerJavaStaticFunction(name: String, valueParameterTypes: Array[Type], returnType: Type, pt: (Type, Seq[SType]) => SType)(cls: Class[_], method: String) {
    registerCode(name, valueParameterTypes, returnType, pt) { case (r, cb, rt, _, args) =>
      val cts = valueParameterTypes.map(PrimitiveTypeToIRIntermediateClassTag(_).runtimeClass)
      val ct = PrimitiveTypeToIRIntermediateClassTag(returnType)
      cb.memoizeAny(
        Code.invokeStatic(cls, method, cts, args.map(a => SType.extractPrimValue(cb, a).get))(ct),
        typeInfoFromClassTag(ct))
    }
  }

  def registerIR(name: String, valueParameterTypes: Array[Type], returnType: Type, inline: Boolean = false, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], Seq[IR], Int) => IR): Unit =
    IRFunctionRegistry.addIR(name, typeParameters, valueParameterTypes, returnType, inline, f)

  def registerSCode1(name: String, mt1: Type, rt: Type, pt: (Type, SType) => SType)(impl: (EmitRegion, EmitCodeBuilder, SType, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1), errorID) => impl(r, cb, rt, a1, errorID)
    }

  def registerSCode1t(name: String, typeParams: Array[Type], mt1: Type, rt: Type, pt: (Type, SType) => SType)(impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1), rt, unwrappedApply(pt), typeParameters = typeParams) {
      case (r, cb, typeParams, rt, Array(a1), errorID) => impl(r, cb, typeParams, rt, a1, errorID)
    }

  def registerSCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2) , errorID) => impl(r, cb, rt, a1, a2, errorID)
    }

  def registerSCode2t(name: String, typeParams: Array[Type], mt1: Type, mt2: Type, rt: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2), rt, unwrappedApply(pt), typeParameters = typeParams) {
      case (r, cb, typeParams, rt, Array(a1, a2), errorID) => impl(r, cb, typeParams, rt, a1, a2, errorID)
    }

  def registerSCode3(name: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, pt: (Type, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SValue, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2, mt3), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3), errorID) => impl(r, cb, rt, a1, a2, a3, errorID)
    }

  def registerSCode4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (Type, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SValue, SValue, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4), errorID) => impl(r, cb, rt, a1, a2, a3, a4, errorID)
    }

  def registerSCode4t(name: String, typeParams: Array[Type], mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type,
    pt: (Type, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, Seq[Type], SType, SValue, SValue, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt), typeParams) {
      case (r, cb, typeParams, rt, Array(a1, a2, a3, a4), errorID) => impl(r, cb, typeParams, rt, a1, a2, a3, a4, errorID)
    }


  def registerSCode5(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, rt: Type, pt: (Type, SType, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SValue, SValue, SValue, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4, mt5), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4, a5), errorID) => impl(r, cb, rt, a1, a2, a3, a4, a5, errorID)
    }

  def registerSCode6(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, mt6: Type, rt: Type, pt: (Type, SType, SType, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SValue, SValue, SValue, SValue, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4, mt5, mt6), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4, a5, a6), errorID) => impl(r, cb, rt, a1, a2, a3, a4, a5, a6, errorID)
    }

  def registerSCode7(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, mt6: Type, mt7: Type, rt: Type, pt: (Type, SType, SType, SType, SType, SType, SType, SType) => SType)
    (impl: (EmitRegion, EmitCodeBuilder, SType, SValue, SValue, SValue, SValue, SValue, SValue, SValue, Value[Int]) => SValue): Unit =
    registerSCode(name, Array(mt1, mt2, mt3, mt4, mt5, mt6, mt7), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4, a5, a6, a7), errorID) => impl(r, cb, rt, a1, a2, a3, a4, a5, a6, a7, errorID)
    }

  def registerCode1(name: String, mt1: Type, rt: Type, pt: (Type, SType) => SType)(impl: (EmitCodeBuilder, EmitRegion, SType, SValue) => Value[_]): Unit =
    registerCode(name, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, cb, rt, _, Array(a1)) => impl(cb, r, rt, a1)
    }

  def registerCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, SType, SType) => SType)
    (impl: (EmitCodeBuilder, EmitRegion, SType, SValue, SValue) => Value[_]): Unit =
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

  def registerIEmitCode3(name: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, pt: (Type, EmitType, EmitType, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1, a2, a3)) =>
      impl(cb, r, rt, errorID, a1, a2, a3)
    }

  def registerIEmitCode4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (Type, EmitType, EmitType, EmitType, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode, EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1, a2, a3, a4)) =>
      impl(cb, r, rt, errorID, a1, a2, a3, a4)
    }

  def registerIEmitCode5(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, rt: Type, pt: (Type, EmitType, EmitType, EmitType, EmitType, EmitType) => EmitType)
    (impl: (EmitCodeBuilder, Value[Region], SType, Value[Int], EmitCode, EmitCode, EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3, mt4, mt5), rt, unwrappedApply(pt)) { case (cb, r, rt, errorID, Array(a1, a2, a3, a4, a5)) =>
      impl(cb, r, rt, errorID, a1, a2, a3, a4, a5)
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

  def apply(r: EmitRegion, cb: EmitCodeBuilder, returnSType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: SValue*): SValue

  def apply(r: EmitRegion, returnType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): EmitCode = {
    EmitCode.fromI(r.mb)(cb => IEmitCode.multiMapEmitCodes(cb, args.toFastSeq) { args =>
      apply(r, cb, returnType, typeParameters, errorID, args: _*)
    })
  }

  def applyI(r: EmitRegion, cb: EmitCodeBuilder, returnType: SType, typeParameters: Seq[Type], errorID: Value[Int], args: EmitCode*): IEmitCode = {
    IEmitCode.multiMapEmitCodes(cb, args.toFastSeq) { args =>
      apply(r, cb, returnType, typeParameters, errorID, args: _*)
    }
  }

  def getAsMethod[C](cb: EmitClassBuilder[C], rpt: SType, typeParameters: Seq[Type], args: SType*): EmitMethodBuilder[C] = {
    val unified = unify(typeParameters, args.map(_.virtualType), rpt.virtualType)
    assert(unified, name)
    val methodbuilder = cb.genEmitMethod(name, FastSeq[ParamType](typeInfo[Region], typeInfo[Int]) ++ args.map(_.paramType), rpt.paramType)
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
