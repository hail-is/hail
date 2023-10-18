package is.hail.expr.ir

import is.hail.annotations.{Region, RegionPool, RegionValueBuilder}
import is.hail.asm4s._
import is.hail.backend.{BackendUtils, ExecuteContext, HailTaskContext}
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.ir.orderings.{CodeOrdering, StructOrdering}
import is.hail.io.fs.FS
import is.hail.io.{BufferSpec, InputBuffer, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.physical.{PCanonicalTuple, PType}
import is.hail.types.virtual.Type
import is.hail.utils._
import is.hail.utils.prettyPrint.ArrayOfByteArrayInputStream
import is.hail.variant.ReferenceGenome
import org.apache.spark.TaskContext

import java.io._
import java.lang.reflect.InvocationTargetException
import scala.collection.mutable
import scala.language.existentials

class EmitModuleBuilder(val ctx: ExecuteContext, val modb: ModuleBuilder) {
  def newEmitClass[C](name: String, sourceFile: Option[String] = None)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] =
    new EmitClassBuilder(this, modb.newClass(name, sourceFile))

  def genEmitClass[C](baseName: String, sourceFile: Option[String] = None)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] =
    newEmitClass[C](genName("C", baseName), sourceFile)

  private[this] val _staticHailClassLoader: StaticField[HailClassLoader] = {
    val cls = genEmitClass[Unit]("HailClassLoaderContainer")
    cls.newStaticField[HailClassLoader]("hailClassLoader", Code._null[HailClassLoader])
  }

  def setHailClassLoader(cb: EmitCodeBuilder, fs: Code[HailClassLoader]): Unit = cb += _staticHailClassLoader.put(fs)

  def getHailClassLoader: Value[HailClassLoader] = new StaticFieldRef(_staticHailClassLoader)

  private[this] val _staticFS: StaticField[FS] = {
    val cls = genEmitClass[Unit]("FSContainer")
    cls.newStaticField[FS]("filesystem", Code._null[FS])
  }

  def setFS(cb: EmitCodeBuilder, fs: Code[FS]): Unit = cb += _staticFS.put(fs)

  def getFS: Value[FS] = new StaticFieldRef(_staticFS)

  private val rgContainers: mutable.Map[String, StaticField[ReferenceGenome]] = mutable.Map.empty

  def hasReferences: Boolean = rgContainers.nonEmpty

  def getReferenceGenome(rg: String): Value[ReferenceGenome] = {
    val rgField = rgContainers.getOrElseUpdate(rg, {
      val cls = genEmitClass[Unit](s"RGContainer_${rg}")
      cls.newStaticField("reference_genome", Code._null[ReferenceGenome])
    })
    new StaticFieldRef(rgField)
  }

  def referenceGenomes(): IndexedSeq[ReferenceGenome] = rgContainers.keys.map(ctx.getReference(_)).toIndexedSeq.sortBy(_.name)
  def referenceGenomeFields(): IndexedSeq[StaticField[ReferenceGenome]] = rgContainers.toFastSeq.sortBy(_._1).map(_._2)

  var _rgMapField: StaticFieldRef[Map[String, ReferenceGenome]] = null

  def referenceGenomeMap: Value[Map[String, ReferenceGenome]] = {
    if (_rgMapField == null) {
      val cls = genEmitClass[Unit](s"RGMapContainer")
      _rgMapField = new StaticFieldRef(cls.newStaticField("reference_genome_map", Code._null[Map[String, ReferenceGenome]]))
    }
    _rgMapField
  }

  def setObjects(cb: EmitCodeBuilder, objects: Code[Array[AnyRef]]): Unit = modb.setObjects(cb, objects)

  def getObject[T <: AnyRef : TypeInfo](obj: T): Code[T] = modb.getObject(obj)

  private[this] var currLitIndex: Int = 0
  private[this] val literalsBuilder = mutable.Map.empty[(VirtualTypeWithReq, Any), (PType, Int)]
  private[this] val encodedLiteralsBuilder = mutable.Map.empty[EncodedLiteral, (PType, Int)]

  def registerLiteral(t: VirtualTypeWithReq, value: Any): (PType, Int) = {
    literalsBuilder.getOrElseUpdate((t, value), {
      val curr = currLitIndex
      val pt = t.canonicalPType
      literalsBuilder.put((t, value), (pt, curr))
      currLitIndex += 1
      (pt, curr)
    })
  }

  def registerEncodedLiteral(el: EncodedLiteral): (PType, Int) = {
    encodedLiteralsBuilder.getOrElseUpdate(el, {
      val curr = currLitIndex
      val pt = el.codec.decodedPType()
      encodedLiteralsBuilder.put(el, (pt, curr))
      currLitIndex += 1
      (pt, curr)
    })
  }

  def literalsResult(): (Array[(VirtualTypeWithReq, Any, PType, Int)], Array[(EncodedLiteral, PType, Int)]) = {
    (literalsBuilder.toArray.map { case ((vt, a), (pt, i)) => (vt, a, pt, i) },
      encodedLiteralsBuilder.toArray.map { case (el, (pt, i)) => (el, pt, i) })
  }

  def hasLiterals: Boolean = currLitIndex > 0
}

trait WrappedEmitModuleBuilder {
  def emodb: EmitModuleBuilder

  def modb: ModuleBuilder = emodb.modb

  def ctx: ExecuteContext = emodb.ctx

  def newEmitClass[C](name: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] = emodb.newEmitClass[C](name)

  def genEmitClass[C](baseName: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] = emodb.genEmitClass[C](baseName)

  def getReferenceGenome(rg: String): Value[ReferenceGenome] = emodb.getReferenceGenome(rg)
}

trait WrappedEmitClassBuilder[C] extends WrappedEmitModuleBuilder {
  def ecb: EmitClassBuilder[C]

  def emodb: EmitModuleBuilder = ecb.emodb

  def cb: ClassBuilder[C] = ecb.cb

  def className: String = ecb.className

  def newField[T: TypeInfo](name: String): Field[T] = ecb.newField[T](name)

  def newStaticField[T: TypeInfo](name: String): StaticField[T] = ecb.newStaticField[T](name)

  def newStaticField[T: TypeInfo](name: String, init: Code[T]): StaticField[T] = ecb.newStaticField[T](name, init)

  def genField[T: TypeInfo](baseName: String): Field[T] = ecb.genField(baseName)

  def genFieldThisRef[T: TypeInfo](name: String = null): ThisFieldRef[T] = ecb.genFieldThisRef[T](name)

  def genLazyFieldThisRef[T: TypeInfo](setup: Code[T], name: String = null): Value[T] = ecb.genLazyFieldThisRef(setup, name)

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): Value[T] = ecb.getOrDefineLazyField(setup, id)

  def newPSettable(sb: SettableBuilder, pt: SType, name: String = null): SSettable = ecb.newPSettable(sb, pt, name)

  def newPField(pt: SType): SSettable = ecb.newPField(pt)

  def newPField(name: String, pt: SType): SSettable = ecb.newPField(name, pt)

  def newEmitField(et: EmitType): EmitSettable = ecb.newEmitField(et.st, et.required)

  def newEmitField(pt: SType, required: Boolean): EmitSettable = ecb.newEmitField(pt, required)

  def newEmitField(name: String, et: EmitType): EmitSettable = ecb.newEmitField(name, et.st, et.required)

  def newEmitField(name: String, pt: SType, required: Boolean): EmitSettable = ecb.newEmitField(name, pt, required)

  def fieldBuilder: SettableBuilder = cb.fieldBuilder

  def result(print: Option[PrintWriter] = None): HailClassLoader => C =
    cb.result(ctx.shouldWriteIRFiles(), print)

  def getHailClassLoader: Code[HailClassLoader] = ecb.getHailClassLoader

  def getFS: Code[FS] = ecb.getFS

  def getTaskContext: Value[HailTaskContext] = ecb.getTaskContext

  def getObject[T <: AnyRef : TypeInfo](obj: T): Code[T] = ecb.getObject(obj)

  def getSerializedAgg(i: Int): Code[Array[Byte]] = ecb.getSerializedAgg(i)

  def setSerializedAgg(i: Int, b: Code[Array[Byte]]): Code[Unit] = ecb.setSerializedAgg(i, b)

  def freeSerializedAgg(i: Int): Code[Unit] = ecb.freeSerializedAgg(i)

  def backend(): Code[BackendUtils] = ecb.backend()

  def addModule(name: String, mod: (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]): Unit =
    ecb.addModule(name, mod)

  def partitionRegion: Settable[Region] = ecb.partitionRegion

  def addLiteral(cb: EmitCodeBuilder, v: Any, t: VirtualTypeWithReq): SValue = ecb.addLiteral(cb, v, t)

  def addEncodedLiteral(cb: EmitCodeBuilder, encodedLiteral: EncodedLiteral) = ecb.addEncodedLiteral(cb, encodedLiteral)

  def getPType[T <: PType : TypeInfo](t: T): Code[T] = ecb.getPType(t)

  def getType[T <: Type : TypeInfo](t: T): Code[T] = ecb.getType(t)

  def newEmitMethod(name: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    ecb.newEmitMethod(name, argsInfo, returnInfo)

  def newEmitMethod(name: String, argsInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]): EmitMethodBuilder[C] =
    ecb.newEmitMethod(name, argsInfo, returnInfo)

  def newStaticEmitMethod(name: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    ecb.newStaticEmitMethod(name, argsInfo, returnInfo)

  def genEmitMethod(baseName: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    ecb.genEmitMethod(baseName, argsInfo, returnInfo)

  def genStaticEmitMethod(baseName: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    ecb.genStaticEmitMethod(baseName, argsInfo, returnInfo)

  def addAggStates(aggSigs: Array[agg.AggStateSig]): agg.TupleAggregatorState = ecb.addAggStates(aggSigs)

  def newRNG(seed: Long): Value[IRRandomness] = ecb.newRNG(seed)

  def getThreefryRNG(): Value[ThreefryRandomEngine] = ecb.getThreefryRNG()

  def resultWithIndex(print: Option[PrintWriter] = None)
  : (HailClassLoader, FS, HailTaskContext, Region) => C =
    ecb.resultWithIndex(print)

  def getOrGenEmitMethod(
    baseName: String, key: Any, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType
  )(body: EmitMethodBuilder[C] => Unit): EmitMethodBuilder[C] = ecb.getOrGenEmitMethod(baseName, key, argsInfo, returnInfo)(body)

  def genEmitMethod[R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[R](baseName)

  def genEmitMethod[A: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A, R](baseName)

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, R](baseName)

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, A3, R](baseName)

  def geEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, A3, A4, R](baseName)

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, A3, A4, A5, R](baseName)

  def openUnbuffered(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    getFS.invoke[String, Boolean, InputStream]("open", path, checkCodec)

  def open(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    Code.newInstance[java.io.BufferedInputStream, InputStream](
      getFS.invoke[String, Boolean, InputStream]("open", path, checkCodec))

  def createUnbuffered(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("create", path)

  def create(path: Code[String]): Code[OutputStream] =
    Code.newInstance[java.io.BufferedOutputStream, OutputStream](
      getFS.invoke[String, OutputStream]("create", path))
}

class EmitClassBuilder[C](
  val emodb: EmitModuleBuilder,
  val cb: ClassBuilder[C]
) extends WrappedEmitModuleBuilder { self =>
  // wrapped ClassBuilder methods
  def className: String = cb.className

  def newField[T: TypeInfo](name: String): Field[T] = cb.newField[T](name)

  def newStaticField[T: TypeInfo](name: String): StaticField[T] = cb.newStaticField[T](name)

  def newStaticField[T: TypeInfo](name: String, init: Code[T]): StaticField[T] = cb.newStaticField[T](name, init)

  def genField[T: TypeInfo](baseName: String): Field[T] = cb.genField(baseName)

  def genFieldThisRef[T: TypeInfo](name: String = null): ThisFieldRef[T] = cb.genFieldThisRef[T](name)

  def genLazyFieldThisRef[T: TypeInfo](setup: Code[T], name: String = null): Value[T] = cb.genLazyFieldThisRef(setup, name)

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): Value[T] = cb.getOrDefineLazyField(setup, id)

  def fieldBuilder: SettableBuilder = cb.fieldBuilder

  def result(writeIRs: Boolean, print: Option[PrintWriter] = None): (HailClassLoader) => C = cb.result(writeIRs, print)

  // EmitClassBuilder methods

  def newPSettable(sb: SettableBuilder, st: SType, name: String = null): SSettable = SSettable(sb, st, name)

  def newPField(st: SType): SSettable = newPSettable(fieldBuilder, st)

  def newPField(name: String, st: SType): SSettable = newPSettable(fieldBuilder, st, name)

  def newEmitField(st: SType, required: Boolean): EmitSettable =
    new EmitSettable(if (required) None else Some(genFieldThisRef[Boolean]("emitfield_missing")), newPField(st))

  def newEmitField(name: String, emitType: EmitType): EmitSettable = newEmitField(name, emitType.st, emitType.required)

  def newEmitField(name: String, st: SType, required: Boolean): EmitSettable =
    new EmitSettable(if (required) None else Some(genFieldThisRef[Boolean](name + "_missing")), newPField(name, st))

  private[this] val typMap: mutable.Map[Type, Value[_ <: Type]] =
    mutable.Map()

  private[this] val pTypeMap: mutable.Map[PType, Value[_ <: PType]] = mutable.Map()

  private[this] type CompareMapKey = (SType, SType)
  private[this] val memoizedComparisons: mutable.Map[CompareMapKey, CodeOrdering] =
    mutable.Map[CompareMapKey, CodeOrdering]()


  def numTypes: Int = typMap.size

  private[this] val decodedLiteralsField = genFieldThisRef[Array[Long]]("decoded_lits")

  def literalsArray(): Value[Array[Long]] = decodedLiteralsField

  def setLiteralsArray(cb: EmitCodeBuilder, arr: Code[Array[Long]]): Unit = cb.assign(decodedLiteralsField, arr)

  lazy val partitionRegion: Settable[Region] = genFieldThisRef[Region]("partitionRegion")
  private[this] lazy val _taskContext: Settable[HailTaskContext] = genFieldThisRef[HailTaskContext]("taskContext")
  private[this] lazy val poolField: Settable[RegionPool] = genFieldThisRef[RegionPool]()

  def addLiteral(cb: EmitCodeBuilder, v: Any, t: VirtualTypeWithReq): SValue = {
    assert(v != null)

    val (pt, i) = emodb.registerLiteral(t, v)
    pt.loadCheapSCode(cb, decodedLiteralsField.get(i))
  }

  def addEncodedLiteral(cb: EmitCodeBuilder, encodedLiteral: EncodedLiteral): SValue = {
    val (pt, i) = emodb.registerEncodedLiteral(encodedLiteral)
    pt.loadCheapSCode(cb, decodedLiteralsField.get(i))
  }

  private[this] def encodeLiterals(): Array[AnyRef] = {
    val (literals, preEncodedLiterals) = emodb.literalsResult()
    val litType = PCanonicalTuple(true, literals.map(_._1.canonicalPType.setRequired(true)): _*)
    val spec = TypedCodecSpec(litType, BufferSpec.wireSpec)

    cb.addInterface(typeInfo[FunctionWithLiterals].iname)
    val mb2 = newEmitMethod("addAndDecodeLiterals", FastSeq[ParamType](typeInfo[Array[AnyRef]]), typeInfo[Unit])

    mb2.voidWithBuilder { cb =>
      val allEncodedFields = mb2.getCodeParam[Array[AnyRef]](1)
      cb.assign(decodedLiteralsField, Code.newArray[Long](literals.length + preEncodedLiterals.length))

      val ib = cb.newLocal[InputBuffer]("ib",
        spec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](Code.checkcast[Array[Byte]](allEncodedFields(0)))))

      val lits = spec.encodedType.buildDecoder(spec.encodedVirtualType, this)
        .apply(cb, partitionRegion, ib)
        .asBaseStruct
      literals.zipWithIndex.foreach { case ((t, _, pt, arrIdx), i) =>
        lits.loadField(cb, i)
          .consume(cb,
            cb._fatal("expect non-missing literals!"),
            { pc =>
              cb += (decodedLiteralsField(arrIdx) = pt.store(cb, partitionRegion, pc, false))
            })
      }
      // Handle the pre-encoded literals, which only need to be decoded.
      preEncodedLiterals.zipWithIndex.foreach { case ((encLit, pt, arrIdx), index) =>
        val spec = encLit.codec
        cb.assign(ib, spec.buildCodeInputBuffer(Code.newInstance[ArrayOfByteArrayInputStream, Array[Array[Byte]]](Code.checkcast[Array[Array[Byte]]](allEncodedFields(index + 1)))))
        val decodedValue = encLit.codec.encodedType.buildDecoder(encLit.typ, this)
          .apply(cb, partitionRegion, ib)
        cb += (decodedLiteralsField(arrIdx) = pt.store(cb, partitionRegion, decodedValue, false))
      }
    }

    val baos = new ByteArrayOutputStream()
    val enc = spec.buildEncoder(ctx, litType)(baos, ctx.theHailClassLoader)
    this.emodb.ctx.r.pool.scopedRegion { region =>
      val rvb = new RegionValueBuilder(ctx.stateManager, region)
      rvb.start(litType)
      rvb.startTuple()
      literals.foreach { case (typ, a, _, _) => rvb.addAnnotation(typ.t, a) }
      rvb.endTuple()
      enc.writeRegionValue(rvb.end())
    }
    enc.flush()
    enc.close()
    Array[AnyRef](baos.toByteArray) ++ preEncodedLiterals.map(_._1.value.ba)
  }

  private[this] var _mods: BoxedArrayBuilder[(String, (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])] = new BoxedArrayBuilder()
  private[this] var _backendField: Settable[BackendUtils] = _

  private[this] var _aggSigs: Array[agg.AggStateSig] = _
  private[this] var _aggRegion: Settable[Region] = _
  private[this] var _aggOff: Settable[Long] = _
  private[this] var _aggState: agg.TupleAggregatorState = _
  private[this] var _nSerialized: Int = 0
  private[this] var _aggSerialized: Settable[Array[Array[Byte]]] = _

  def addAggStates(aggSigs: Array[agg.AggStateSig]): agg.TupleAggregatorState = {
    if (_aggSigs != null) {
      assert(aggSigs sameElements _aggSigs)
      return _aggState
    }
    cb.addInterface(typeInfo[FunctionWithAggRegion].iname)
    _aggSigs = aggSigs
    _aggRegion = genFieldThisRef[Region]("agg_top_region")
    _aggOff = genFieldThisRef[Long]("agg_off")
    _aggSerialized = genFieldThisRef[Array[Array[Byte]]]("agg_serialized")

    val newF = newEmitMethod("newAggState", FastSeq[ParamType](typeInfo[Region]), typeInfo[Unit])
    val setF = newEmitMethod("setAggState", FastSeq[ParamType](typeInfo[Region], typeInfo[Long]), typeInfo[Unit])
    val getF = newEmitMethod("getAggOffset", FastSeq[ParamType](), typeInfo[Long])
    val storeF = newEmitMethod("storeAggsToRegion", FastSeq[ParamType](), typeInfo[Unit])
    val setNSer = newEmitMethod("setNumSerialized", FastSeq[ParamType](typeInfo[Int]), typeInfo[Unit])
    val setSer = newEmitMethod("setSerializedAgg", FastSeq[ParamType](typeInfo[Int], typeInfo[Array[Byte]]), typeInfo[Unit])
    val getSer = newEmitMethod("getSerializedAgg", FastSeq[ParamType](typeInfo[Int]), typeInfo[Array[Byte]])

    val (nfcode, states) = EmitCodeBuilder.scoped(newF) { cb =>
      val states = agg.StateTuple(aggSigs.map(a => agg.AggStateSig.getState(a, cb.emb.ecb)).toArray)
      _aggState = new agg.TupleAggregatorState(this, states, _aggRegion, _aggOff)
      cb += (_aggRegion := newF.getCodeParam[Region](1))
      cb += _aggState.topRegion.setNumParents(aggSigs.length)
      cb += (_aggOff := _aggRegion.load().allocate(states.storageType.alignment, states.storageType.byteSize))
      states.createStates(cb)
      _aggState.newState(cb)

      states
    }

    newF.emit(nfcode)

    setF.emitWithBuilder { cb =>
      cb += (_aggRegion := setF.getCodeParam[Region](1))
      cb += _aggState.topRegion.setNumParents(aggSigs.length)
      states.createStates(cb)
      cb += (_aggOff := setF.getCodeParam[Long](2))
      _aggState.load(cb)
      Code._empty
    }

    getF.emitWithBuilder { cb =>
      storeF.invokeCode[Unit](cb)
      _aggOff
    }

    storeF.voidWithBuilder { cb =>
      _aggState.store(cb)
    }

    setNSer.emit(_aggSerialized := Code.newArray[Array[Byte]](setNSer.getCodeParam[Int](1)))

    setSer.emit(_aggSerialized.load().update(setSer.getCodeParam[Int](1), setSer.getCodeParam[Array[Byte]](2)))

    getSer.emit(_aggSerialized.load()(getSer.getCodeParam[Int](1)))

    _aggState
  }

  def getSerializedAgg(i: Int): Code[Array[Byte]] = {
    if (_nSerialized <= i)
      _nSerialized = i + 1
    _aggSerialized.load()(i)
  }

  def setSerializedAgg(i: Int, b: Code[Array[Byte]]): Code[Unit] = {
    if (_nSerialized <= i)
      _nSerialized = i + 1
    _aggSerialized.load().update(i, b)
  }

  def freeSerializedAgg(i: Int): Code[Unit] = {
    assert(i < _nSerialized)
    _aggSerialized.load().update(i, Code._null[Array[Byte]])
  }

  def runMethodWithHailExceptionHandler(mname: String): Code[(String, java.lang.Integer)] = {
    Code.invokeScalaObject2[AnyRef, String, (String, java.lang.Integer)](CodeExceptionHandler.getClass,
      "handleUserException",
      cb._this.get.asInstanceOf[Code[AnyRef]], mname)
  }

  def backend(): Code[BackendUtils] = {
    if (_backendField == null) {
      cb.addInterface(typeInfo[FunctionWithBackend].iname)
      val backendField = genFieldThisRef[BackendUtils]()
      val mb = newEmitMethod("setBackend", FastSeq[ParamType](typeInfo[BackendUtils]), typeInfo[Unit])
      mb.emit(backendField := mb.getCodeParam[BackendUtils](1))
      _backendField = backendField
    }
    _backendField
  }

  def pool(): Value[RegionPool] = {
    poolField
  }

  def addModule(name: String, mod: (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]): Unit = {
    _mods += name -> mod
  }

  def getHailClassLoader: Code[HailClassLoader] = emodb.getHailClassLoader

  def getFS: Code[FS] = emodb.getFS

  def getTaskContext: Value[HailTaskContext] = _taskContext

  def setObjects(cb: EmitCodeBuilder, objects: Code[Array[AnyRef]]): Unit = modb.setObjects(cb, objects)

  def getObject[T <: AnyRef : TypeInfo](obj: T): Code[T] = modb.getObject(obj)

  def makeAddObjects(): Array[AnyRef] = {
    if (emodb.modb._objects == null)
      null
    else {
      cb.addInterface(typeInfo[FunctionWithObjects].iname)
      val mb = newEmitMethod("setObjects", FastSeq[ParamType](typeInfo[Array[AnyRef]]), typeInfo[Unit])
      mb.voidWithBuilder(cb => emodb.setObjects(cb, mb.getCodeParam[Array[AnyRef]](1)))
      emodb.modb._objects.result()
    }
  }

  def getPType[T <: PType : TypeInfo](t: T): Code[T] = emodb.getObject(t)

  def getType[T <: Type : TypeInfo](t: T): Code[T] = emodb.getObject(t)

  def getOrdering(t1: SType,
    t2: SType,
    sortOrder: SortOrder = Ascending
  ): CodeOrdering = {
    val baseOrd = memoizedComparisons.getOrElseUpdate((t1, t2), {
      CodeOrdering.makeOrdering(t1, t2, this)
    })
    sortOrder match {
      case Ascending => baseOrd
      case Descending => baseOrd.reverse
    }
  }

  def getOrderingFunction(
    t1: SType,
    t2: SType,
    sortOrder: SortOrder,
    op: CodeOrdering.Op
  ): CodeOrdering.F[op.ReturnType] = {
    val ord = getOrdering(t1, t2, sortOrder)

    { (cb: EmitCodeBuilder, v1: EmitValue, v2: EmitValue) =>

      val r: Code[_] = op match {
        case CodeOrdering.Compare(missingEqual) => ord.compare(cb, v1, v2, missingEqual)
        case CodeOrdering.Equiv(missingEqual) => ord.equiv(cb, v1, v2, missingEqual)
        case CodeOrdering.Lt(missingEqual) => ord.lt(cb, v1, v2, missingEqual)
        case CodeOrdering.Lteq(missingEqual) => ord.lteq(cb, v1, v2, missingEqual)
        case CodeOrdering.Gt(missingEqual) => ord.gt(cb, v1, v2, missingEqual)
        case CodeOrdering.Gteq(missingEqual) => ord.gteq(cb, v1, v2, missingEqual)
        case CodeOrdering.Neq(missingEqual) => !ord.equiv(cb, v1, v2, missingEqual)
      }
      cb.memoize[op.ReturnType](coerce[op.ReturnType](r))(op.rtti, implicitly[op.ReturnType =!= Unit])
    }
  }

  def getStructOrderingFunction(
    t1: SBaseStruct,
    t2: SBaseStruct,
    sortFields: Array[SortField],
    op: CodeOrdering.Op
  ): CodeOrdering.F[op.ReturnType] = {
    { (cb: EmitCodeBuilder, v1: EmitValue, v2: EmitValue) =>
      val ord = StructOrdering.make(t1, t2, cb.emb.ecb, sortFields.map(_.sortOrder))

      val r: Code[_] = op match {
        case CodeOrdering.StructLt(missingEqual) => ord.lt(cb, v1, v2, missingEqual)
        case CodeOrdering.StructLteq(missingEqual) => ord.lteq(cb, v1, v2, missingEqual)
        case CodeOrdering.StructGt(missingEqual) => ord.gt(cb, v1, v2, missingEqual)
        case CodeOrdering.StructGteq(missingEqual) => ord.gteq(cb, v1, v2, missingEqual)
        case CodeOrdering.StructCompare(missingEqual) => ord.compare(cb, v1, v2, missingEqual)
      }
      cb.memoize[op.ReturnType](coerce[op.ReturnType](r))(op.rtti, implicitly[op.ReturnType =!= Unit])
    }
  }

  // derived functions
  def getOrderingFunction(t: SType, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
    getOrderingFunction(t, t, sortOrder = Ascending, op)

  def getOrderingFunction(t1: SType, t2: SType, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
    getOrderingFunction(t1, t2, sortOrder = Ascending, op)

  def getOrderingFunction(
    t: SType,
    op: CodeOrdering.Op,
    sortOrder: SortOrder
  ): CodeOrdering.F[op.ReturnType] =
    getOrderingFunction(t, t, sortOrder, op)

  private def getCodeArgsInfo(argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): (IndexedSeq[TypeInfo[_]], TypeInfo[_], AsmTuple[_]) = {
    val codeArgsInfo = argsInfo.flatMap {
      case CodeParamType(ti) => FastSeq(ti)
      case t: EmitParamType => t.valueTupleTypes
      case SCodeParamType(pt) => pt.settableTupleTypes()
    }
    val (codeReturnInfo, asmTuple) = returnInfo match {
      case CodeParamType(ti) => ti -> null
      case SCodeParamType(pt) if pt.nSettables == 1 => pt.settableTupleTypes().head -> null
      case SCodeParamType(pt) =>
        val asmTuple = modb.tupleClass(pt.settableTupleTypes())
        asmTuple.ti -> asmTuple
      case t: EmitParamType =>
        val ts = t.valueTupleTypes
        if (ts.length == 1)
          ts.head -> null
        else {
          val asmTuple = modb.tupleClass(ts)
          asmTuple.ti -> asmTuple
        }
    }

    (codeArgsInfo, codeReturnInfo, asmTuple)
  }

  def newEmitMethod(name: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] = {
    val (codeArgsInfo, codeReturnInfo, asmTuple) = getCodeArgsInfo(argsInfo, returnInfo)

    new EmitMethodBuilder[C](argsInfo, returnInfo, this, cb.newMethod(name, codeArgsInfo, codeReturnInfo), asmTuple)
  }

  def newEmitMethod(name: String, argsInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]): EmitMethodBuilder[C] = {
    new EmitMethodBuilder[C](
      argsInfo.map(ai => CodeParamType(ai.base)), CodeParamType(returnInfo.base),
      this, cb.newMethod(name, argsInfo, returnInfo), asmTuple = null)
  }

  def newStaticEmitMethod(name: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] = {
    val (codeArgsInfo, codeReturnInfo, asmTuple) = getCodeArgsInfo(argsInfo, returnInfo)

    new EmitMethodBuilder[C](argsInfo, returnInfo, this,
      cb.newStaticMethod(name, codeArgsInfo, codeReturnInfo),
      asmTuple)
  }

  val rngs: BoxedArrayBuilder[(Settable[IRRandomness], Code[IRRandomness])] = new BoxedArrayBuilder()

  var threefryRNG: Option[(Settable[ThreefryRandomEngine], Code[ThreefryRandomEngine])] = None

  def makeAddPartitionRegion(): Unit = {
    cb.addInterface(typeInfo[FunctionWithPartitionRegion].iname)
    val mb = newEmitMethod("addPartitionRegion", FastSeq[ParamType](typeInfo[Region]), typeInfo[Unit])
    mb.emit(partitionRegion := mb.getCodeParam[Region](1))
    val mb2 = newEmitMethod("setPool", FastSeq[ParamType](typeInfo[RegionPool]), typeInfo[Unit])
    mb2.emit(poolField := mb2.getCodeParam[RegionPool](1))
  }

  def makeAddHailClassLoader(): Unit = {
    cb.addInterface(typeInfo[FunctionWithHailClassLoader].iname)
    val mb = newEmitMethod("addHailClassLoader", FastSeq[ParamType](typeInfo[HailClassLoader]), typeInfo[Unit])
    mb.voidWithBuilder { cb =>
      emodb.setHailClassLoader(cb, mb.getCodeParam[HailClassLoader](1))
    }
  }

  def makeAddFS(): Unit = {
    cb.addInterface(typeInfo[FunctionWithFS].iname)
    val mb = newEmitMethod("addFS", FastSeq[ParamType](typeInfo[FS]), typeInfo[Unit])
    mb.voidWithBuilder { cb =>
      emodb.setFS(cb, mb.getCodeParam[FS](1))
    }
  }

  def makeAddTaskContext(): Unit = {
    cb.addInterface(typeInfo[FunctionWithTaskContext].iname)
    val mb = newEmitMethod("addTaskContext", FastSeq[ParamType](typeInfo[HailTaskContext]), typeInfo[Unit])
    mb.voidWithBuilder { cb =>
      cb.assign(_taskContext, mb.getCodeParam[HailTaskContext](1))
    }
  }

  def makeAddReferenceGenomes(): Unit = {
    cb.addInterface(typeInfo[FunctionWithReferences].iname)
    val mb = newEmitMethod("addReferenceGenomes", FastSeq[ParamType](typeInfo[Array[ReferenceGenome]]), typeInfo[Unit])
    mb.voidWithBuilder { cb =>
      val rgFields = emodb.referenceGenomeFields()
      val rgs = mb.getCodeParam[Array[ReferenceGenome]](1)
      cb.if_(rgs.length().cne(const(rgFields.length)), cb._fatal("Invalid number of references, expected ", rgFields.length.toString, " got ", rgs.length().toS))
      for ((fld, i) <- rgFields.zipWithIndex) {
        cb += rgs(i).invoke[String, FS, Unit]("heal", ctx.localTmpdir, getFS)
        cb += fld.put(rgs(i))
      }

      Option(emodb._rgMapField).foreach { fld =>
        cb.assign(fld, Code.invokeStatic1[ReferenceGenome, Array[ReferenceGenome], Map[String, ReferenceGenome]]("getMapFromArray", rgs))
      }
    }
  }

  def makeRNGs(): Unit = {
    cb.addInterface(typeInfo[FunctionWithSeededRandomness].iname)

    val initialized = genFieldThisRef[Boolean]("initialized")
    val mb = newEmitMethod("setPartitionIndex", IndexedSeq[ParamType](typeInfo[Int]), typeInfo[Unit])
    val rngFields = rngs.result()

    mb.voidWithBuilder { cb =>
      cb.if_(!initialized, {
        rngFields.foreach { case (field, init) =>
          cb.assign(field, init)
        }

        threefryRNG.foreach { case (field, init) =>
          cb.assign(field, init)
        }

        cb.assign(initialized, true)
      })

      rngFields.foreach { case (field, _) =>
        cb += field.invoke[Int, Unit]("reset", mb.getCodeParam[Int](1))
      }
    }
  }

  def newRNG(seed: Long): Value[IRRandomness] = {
    val rng = genFieldThisRef[IRRandomness]()
    rngs += rng -> Code.newInstance[IRRandomness, Long](seed)
    rng
  }

  def getThreefryRNG(): Value[ThreefryRandomEngine] = {
    threefryRNG match {
      case Some((rngField, _)) => rngField
      case None =>
        val rngField = genFieldThisRef[ThreefryRandomEngine]()
        val rngInit = Code.invokeScalaObject0[ThreefryRandomEngine](
          ThreefryRandomEngine.getClass, "apply")
        threefryRNG = Some(rngField -> rngInit)
        rngField
    }
  }

  def resultWithIndex(print: Option[PrintWriter] = None)
  : (HailClassLoader, FS, HailTaskContext, Region) => C = {
    makeRNGs()
    makeAddPartitionRegion()
    makeAddHailClassLoader()
    makeAddFS()
    makeAddTaskContext()

    val hasLiterals: Boolean = emodb.hasLiterals
    val hasReferences: Boolean = emodb.hasReferences
    if (hasReferences)
      makeAddReferenceGenomes()

    val objects = makeAddObjects()

    val literalsBc = if (hasLiterals)
      ctx.backend.broadcast(encodeLiterals())
    else
      // if there are no literals, there might not be a HailContext
      null

    val references: Array[ReferenceGenome] = if (hasReferences)
      emodb.referenceGenomes().toArray
    else
      null


    val nSerializedAggs = _nSerialized

    val useBackend = _backendField != null
    val backend = if (useBackend) new BackendUtils(_mods.result()) else null

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    val n = cb.className.replace("/", ".")
    val classesBytes = modb.classesBytes(ctx.shouldWriteIRFiles(), print)

    new ((HailClassLoader, FS, HailTaskContext, Region) => C) with java.io.Serializable {
      @transient @volatile private var theClass: Class[_] = null

      def apply(hcl: HailClassLoader, fs: FS, htc: HailTaskContext, region: Region): C = {
        if (theClass == null) {
          this.synchronized {
            if (theClass == null) {
              classesBytes.load(hcl)
              theClass = loadClass(hcl, n)
            }
          }
        }
        val idx = htc.partitionId()

        val f = theClass.getDeclaredConstructor().newInstance().asInstanceOf[C]
        f.asInstanceOf[FunctionWithHailClassLoader].addHailClassLoader(hcl)
        f.asInstanceOf[FunctionWithFS].addFS(fs)
        f.asInstanceOf[FunctionWithTaskContext].addTaskContext(htc)
        f.asInstanceOf[FunctionWithPartitionRegion].addPartitionRegion(region)
        f.asInstanceOf[FunctionWithPartitionRegion].setPool(region.pool)
        if (useBackend)
          f.asInstanceOf[FunctionWithBackend].setBackend(backend)
        if (objects != null)
          f.asInstanceOf[FunctionWithObjects].setObjects(objects)
        if (hasLiterals)
          f.asInstanceOf[FunctionWithLiterals].addAndDecodeLiterals(literalsBc.value)
        if (hasReferences)
          f.asInstanceOf[FunctionWithReferences].addReferenceGenomes(references)
        if (nSerializedAggs != 0)
          f.asInstanceOf[FunctionWithAggRegion].setNumSerialized(nSerializedAggs)
        f.asInstanceOf[FunctionWithSeededRandomness].setPartitionIndex(idx)
        f
      }
    }
  }

  private[this] val methodMemo: mutable.Map[Any, EmitMethodBuilder[C]] = mutable.Map()

  def getOrGenEmitMethod(
    baseName: String, key: Any, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType
  )(body: EmitMethodBuilder[C] => Unit): EmitMethodBuilder[C] = {
    methodMemo.getOrElse(key, {
      val mb = genEmitMethod(baseName, argsInfo, returnInfo)
      methodMemo(key) = mb
      body(mb)
      mb
    })
  }

  def genEmitMethod(baseName: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    newEmitMethod(genName("m", baseName), argsInfo, returnInfo)

  def genEmitMethod[R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastSeq[ParamType](), typeInfo[R])

  def genEmitMethod[A: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastSeq[ParamType](typeInfo[A]), typeInfo[R])

  def genEmitMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastSeq[ParamType](typeInfo[A], typeInfo[B]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastSeq[ParamType](typeInfo[A1], typeInfo[A2], typeInfo[A3]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastSeq[ParamType](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastSeq[ParamType](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4], typeInfo[A5]), typeInfo[R])

  def genStaticEmitMethod(baseName: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    newStaticEmitMethod(genName("sm", baseName), argsInfo, returnInfo)

  def getUnsafeReader(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    getFS.invoke[String, Boolean, InputStream]("unsafeReader", path, checkCodec)

  def getUnsafeWriter(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("unsafeWriter", path)
}

object EmitFunctionBuilder {
  def apply[F](
    ctx: ExecuteContext, baseName: String, paramTypes: IndexedSeq[ParamType], returnType: ParamType, sourceFile: Option[String] = None
  )(implicit fti: TypeInfo[F]): EmitFunctionBuilder[F] = {
    val modb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val cb = modb.genEmitClass[F](baseName, sourceFile)
    val apply = cb.newEmitMethod("apply", paramTypes, returnType)
    new EmitFunctionBuilder(apply)
  }

  def apply[F](
    ctx: ExecuteContext, baseName: String, argInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]
  )(implicit fti: TypeInfo[F]): EmitFunctionBuilder[F] = {
    val modb = new EmitModuleBuilder(ctx, new ModuleBuilder())
    val cb = modb.genEmitClass[F](baseName)
        val apply = cb.newEmitMethod("apply", argInfo, returnInfo)
    new EmitFunctionBuilder(apply)
  }

  def apply[R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction0[R]] =
    EmitFunctionBuilder[AsmFunction0[R]](ctx, baseName, FastSeq[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def apply[A: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction1[A, R]] =
    EmitFunctionBuilder[AsmFunction1[A, R]](ctx, baseName, Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction2[A, B, R]] =
    EmitFunctionBuilder[AsmFunction2[A, B, R]](ctx, baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction3[A, B, C, R]] =
    EmitFunctionBuilder[AsmFunction3[A, B, C, R]](ctx, baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]](ctx, baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]](ctx, baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]](ctx, baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo](ctx: ExecuteContext, baseName: String): EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]](ctx, baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
}

trait FunctionWithObjects {
  def setObjects(objects: Array[AnyRef]): Unit
}

trait FunctionWithAggRegion {
  // Calls storeAggsToRegion, and returns the aggregator state offset in the top agg region
  def getAggOffset(): Long

  // stores agg regions into the top agg region, so that all agg resources are referenced solely by that region
  def storeAggsToRegion(): Unit

  // Sets the function's agg container to the agg state at $offset, loads agg regions onto class
  def setAggState(region: Region, offset: Long): Unit

  def newAggState(region: Region): Unit

  def setNumSerialized(i: Int): Unit

  def setSerializedAgg(i: Int, b: Array[Byte]): Unit

  def getSerializedAgg(i: Int): Array[Byte]
}

trait FunctionWithHailClassLoader {
  def addHailClassLoader(fs: HailClassLoader): Unit
}

trait FunctionWithFS {
  def addFS(fs: FS): Unit
}

trait FunctionWithTaskContext {
  def addTaskContext(htc: HailTaskContext): Unit
}

trait FunctionWithReferences {
  def addReferenceGenomes(rgs: Array[ReferenceGenome]): Unit
}

trait FunctionWithPartitionRegion {
  def addPartitionRegion(r: Region): Unit
  def setPool(pool: RegionPool): Unit
}

trait FunctionWithLiterals {
  def addAndDecodeLiterals(lit: Array[AnyRef]): Unit
}

trait FunctionWithSeededRandomness {
  def setPartitionIndex(idx: Int): Unit
}

trait FunctionWithBackend {
  def setBackend(spark: BackendUtils): Unit
}

object CodeExceptionHandler {
  /**
    * This method assumes that the method referred to by `methodName`
    * is a -argument class method (only takes the class itself as an arg)
    * which returns void.
    */
  def handleUserException(obj: AnyRef, methodName: String): (String, java.lang.Integer) = {
    try {
      obj.getClass.getMethod(methodName).invoke(obj)
      null
    } catch {
      case e: InvocationTargetException =>
        e.getTargetException match {
          case ue: HailException => (ue.msg, ue.errorId)
          case e => throw e
        }
    }
  }
}

class EmitMethodBuilder[C](
  val emitParamTypes: IndexedSeq[ParamType],
  val emitReturnType: ParamType,
  val ecb: EmitClassBuilder[C],
  val mb: MethodBuilder[C],
  private[ir] val asmTuple: AsmTuple[_]
) extends WrappedEmitClassBuilder[C] {
  private[this] val nCodeArgs = emitParamTypes.map(_.nCodes).sum
  if (nCodeArgs > 255)
    throw new RuntimeException(s"invalid method ${ mb.methodName }: ${ nCodeArgs } code arguments:" +
      s"\n  ${ emitParamTypes.map(p => s"${ p.nCodes } - $p").mkString("\n  ") }")
  // wrapped MethodBuilder methods
  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] = mb.newLocal[T](name)

  def localBuilder: SettableBuilder = mb.localBuilder

  // FIXME needs to code and emit variants
  def emit(body: Code[_]): Unit = mb.emit(body)

  // EmitMethodBuilder methods

  // this, ...
  private val emitParamCodeIndex = emitParamTypes.scanLeft((!mb.isStatic).toInt) {
    case (i, paramType) =>
      i + paramType.nCodes
  }

  def getCodeParam[T: TypeInfo](emitIndex: Int): Settable[T] = {
    if (emitIndex == 0 && !mb.isStatic)
      mb.getArg[T](0)
    else {
      val static = (!mb.isStatic).toInt
      assert(emitParamTypes(emitIndex - static).isInstanceOf[CodeParamType])
      mb.getArg[T](emitParamCodeIndex(emitIndex - static))
    }
  }

  def getSCodeParam(emitIndex: Int): SValue = {
    assert(mb.isStatic || emitIndex != 0)
    val static = (!mb.isStatic).toInt
    val _st = emitParamTypes(emitIndex - static).asInstanceOf[SCodeParamType].st
    assert(_st.isRealizable)

    val ts = _st.settableTupleTypes()
    val codeIndex = emitParamCodeIndex(emitIndex - static)

    _st.fromValues(ts.zipWithIndex.map { case (t, i) =>
      mb.getArg(codeIndex + i)(t)
    })
  }

  def storeEmitParamAsField(cb: EmitCodeBuilder, emitIndex: Int): EmitValue = {
    val param = getEmitParam(cb, emitIndex)
    val fd = newEmitField(s"${mb.methodName}_param_${emitIndex}", param.emitType)
    cb.assign(fd, param)
    fd
  }

  def getEmitParam(cb: EmitCodeBuilder, emitIndex: Int): EmitValue = {
    assert(mb.isStatic || emitIndex != 0)
    val static = (!mb.isStatic).toInt
    val et = emitParamTypes(emitIndex - static) match {
      case t: EmitParamType => t
      case _ => throw new RuntimeException(s"isStatic=${ mb.isStatic }, emitIndex=$emitIndex, params=$emitParamTypes")
    }
    val codeIndex = emitParamCodeIndex(emitIndex - static)

    et match {
      case SingleCodeEmitParamType(required, sct) =>
        val m = if (required) None else Some(mb.getArg[Boolean](codeIndex + 1))
        val v = sct.loadToSValue(cb, mb.getArg(codeIndex)(sct.ti))

        EmitValue(m, v)

      case SCodeEmitParamType(et) =>
        val ts = et.st.settableTupleTypes()

        val m = if (et.required) None else Some(mb.getArg[Boolean](codeIndex + ts.length))
        val v = et.st.fromValues(ts.zipWithIndex.map { case (t, i) =>
          mb.getArg(codeIndex + i)(t)
        })
        EmitValue(m, v)
    }
  }

  def invokeCode[T](cb: CodeBuilderLike, args: Param*): Value[T] = {
    assert(emitReturnType.isInstanceOf[CodeParamType])
    assert(args.forall(_.isInstanceOf[CodeParam]))
    mb.invoke(cb, args.flatMap {
      case CodeParam(c) => FastSeq(c)
      // If you hit this assertion, it means that an EmitParam was passed to
      // invokeCode. Code with EmitParams must be invoked using the EmitCodeBuilder
      // interface to ensure that setup is run and missingness is evaluated for the
      // EmitCode
      case EmitParam(ec) => fatal("EmitParam passed to invokeCode")
    }: _*)
  }
  def newPLocal(st: SType): SSettable = newPSettable(localBuilder, st)

  def newPLocal(name: String, st: SType): SSettable = newPSettable(localBuilder, st, name)

  def newEmitLocal(emitType: EmitType): EmitSettable = newEmitLocal(emitType.st, emitType.required)
  def newEmitLocal(st: SType, required: Boolean): EmitSettable =
    new EmitSettable(if (required) None else Some(newLocal[Boolean]("anon_emitlocal_m")), newPLocal("anon_emitlocal_v", st))

  def newEmitLocal(name: String, emitType: EmitType): EmitSettable = newEmitLocal(name, emitType.st, emitType.required)
  def newEmitLocal(name: String, st: SType, required: Boolean): EmitSettable =
    new EmitSettable(if (required) None else Some(newLocal[Boolean](name + "_missing")), newPLocal(name, st))

  def emitWithBuilder[T](f: (EmitCodeBuilder) => Code[T]): Unit = emit(EmitCodeBuilder.scopedCode[T](this)(f))

  def voidWithBuilder(f: (EmitCodeBuilder) => Unit): Unit = emit(EmitCodeBuilder.scopedVoid(this)(f))

  def emitSCode(f: (EmitCodeBuilder) => SValue): Unit = {
    emit(EmitCodeBuilder.scopedCode(this) { cb =>
      val res = f(cb)
      if (res.st.nSettables == 1)
        res.valueTuple.head
      else
        asmTuple.newTuple(res.valueTuple.map(_.get))
    })
  }

  def implementLabel(label: CodeLabel)(f: EmitCodeBuilder => Unit): Unit = {
    EmitCodeBuilder.scopedVoid(this) { cb =>
      cb.define(label)
      f(cb)
      // assert(!cb.isOpenEnded)
        /*
        FIXME: The above assertion should hold, but currently does not. This is
        likely due to client code with patterns like the following, which incorrectly
        leaves the code builder open-ended:

        cb.ifx(b,
          cb.goto(L1),
          cb.goto(L2))
         */
    }
  }

  def defineAndImplementLabel(f: EmitCodeBuilder => Unit): CodeLabel = {
    val label = CodeLabel()
    implementLabel(label)(f)
    label
  }

  override def toString: String = {
    s"[${ mb.methodName }]${ super.toString }"
  }
}

trait WrappedEmitMethodBuilder[C] extends WrappedEmitClassBuilder[C] {
  def emb: EmitMethodBuilder[C]

  def ecb: EmitClassBuilder[C] = emb.ecb

  // wrapped MethodBuilder methods
  def mb: MethodBuilder[C] = emb.mb

  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] = mb.newLocal[T](name)

  def localBuilder = mb.localBuilder

  // FIXME needs to code and emit variants
  def emit(body: Code[_]): Unit = mb.emit(body)

  def emitWithBuilder[T](f: (EmitCodeBuilder) => Code[T]): Unit = emb.emitWithBuilder(f)

  // EmitMethodBuilder methods
  def getCodeParam[T: TypeInfo](emitIndex: Int): Settable[T] = emb.getCodeParam[T](emitIndex)

  def getEmitParam(cb: EmitCodeBuilder, emitIndex: Int): EmitValue = emb.getEmitParam(cb, emitIndex)

  def newPLocal(st: SType): SSettable = emb.newPLocal(st)

  def newPLocal(name: String, st: SType): SSettable = emb.newPLocal(name, st)

  def newEmitLocal(st: SType, required: Boolean): EmitSettable = emb.newEmitLocal(st, required)

  def newEmitLocal(name: String, pt: SType, required: Boolean): EmitSettable = emb.newEmitLocal(name, pt, required)
}

class EmitFunctionBuilder[F](val apply_method: EmitMethodBuilder[F]) extends WrappedEmitMethodBuilder[F] {
  def emb: EmitMethodBuilder[F] = apply_method
}
