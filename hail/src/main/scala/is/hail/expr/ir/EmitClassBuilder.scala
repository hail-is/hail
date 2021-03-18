package is.hail.expr.ir

import java.io._
import java.util.Base64

import is.hail.{HailContext, lir}
import is.hail.annotations.{Region, RegionPool, RegionValueBuilder, SafeRow}
import is.hail.asm4s._
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.backend.BackendUtils
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.io.fs.FS
import is.hail.io.{BufferSpec, InputBuffer, TypedCodecSpec}
import is.hail.lir
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.{PBaseStructValue, PCanonicalTuple, PCode, PSettable, PStream, PStruct, PType, PValue, typeToTypeInfo}
import is.hail.types.virtual.Type
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.TaskContext

import scala.collection.mutable
import scala.language.existentials

class EmitModuleBuilder(val ctx: ExecuteContext, val modb: ModuleBuilder) {
  def newEmitClass[C](name: String, sourceFile: Option[String] = None)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] =
    new EmitClassBuilder(this, modb.newClass(name, sourceFile))

  def genEmitClass[C](baseName: String, sourceFile: Option[String] = None)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] =
    newEmitClass[C](genName("C", baseName), sourceFile)

  private[this] var _staticFS: Settable[FS] = _

  def getFS: Value[FS] = {
    if (_staticFS == null) {
      val fsinfo = typeInfoFromClass(ctx.fs.getClass)
      val cls = genEmitClass[Unit]("FSContainer")
      val baos = new ByteArrayOutputStream()
      val oos = new ObjectOutputStream(baos)
      oos.writeObject(ctx.fs)

      val fsbytes = baos.toByteArray()
      val fsstring = Base64.getEncoder().encodeToString(fsbytes)

      val chunkSize = (1 << 16) - 1
      val nChunks = (fsstring.length() - 1) / chunkSize + 1
      assert(nChunks > 0)

      val chunks = Array.tabulate(nChunks){ i => fsstring.slice(i * chunkSize, (i + 1) * chunkSize) }
      val stringAssembler =
        chunks.tail.foldLeft[Code[String]](chunks.head) { (c, s) => c.invoke[String, String]("concat", s) }

      val mb = cls.newStaticEmitMethod("init_filesystem", FastIndexedSeq(), typeInfo[FS])
      mb.emitWithBuilder { cb =>
        val b64 = Code.invokeStatic0[Base64, Base64.Decoder]("getDecoder")
        val ba = b64.invoke[String, Array[Byte]]("decode", stringAssembler)
        val bais = Code.newInstance[ByteArrayInputStream, Array[Byte]](ba)
        val ois = cb.newLocal("ois", Code.newInstance[ObjectInputStream, InputStream](bais))
        Code.checkcast(ois.invoke[Any]("readObject"))(fsinfo)
      }
      val fs = cls.newStaticField[FS]("filesystem", mb.invokeCode())

      _staticFS = new StaticFieldRef(fs)
    }
    _staticFS
  }

  private[this] val rgMap: mutable.Map[ReferenceGenome, Value[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Value[ReferenceGenome]]()

  def getReferenceGenome(rg: ReferenceGenome): Value[ReferenceGenome] = rgMap.getOrElseUpdate(rg, {
    val cls = genEmitClass[Unit](s"RGContainer_${rg.name}")
    val fld = cls.newStaticField("reference_genome", rg.codeSetup(ctx.localTmpdir, cls))
    new StaticFieldRef(fld)
  })
}

trait WrappedEmitModuleBuilder {
  def emodb: EmitModuleBuilder

  def modb: ModuleBuilder = emodb.modb

  def ctx: ExecuteContext = emodb.ctx

  def newEmitClass[C](name: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] = emodb.newEmitClass[C](name)

  def genEmitClass[C](baseName: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] = emodb.genEmitClass[C](baseName)

  def getReferenceGenome(rg: ReferenceGenome): Value[ReferenceGenome] = emodb.getReferenceGenome(rg)
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

  def newPSettable(sb: SettableBuilder, pt: PType, name: String = null): PSettable = ecb.newPSettable(sb, pt, name)

  def newPField(pt: PType): PSettable = ecb.newPField(pt)

  def newPField(name: String, pt: PType): PSettable = ecb.newPField(name, pt)

  def newEmitField(pt: PType): EmitSettable = ecb.newEmitField(pt)

  def newEmitField(name: String, pt: PType): EmitSettable = ecb.newEmitField(name, pt)

  def newEmitSettable(pt: PType, ms: Settable[Boolean], vs: PSettable): EmitSettable = ecb.newEmitSettable(pt, ms, vs)

  def newPresentEmitField(pt: PType): PresentEmitSettable = ecb.newPresentEmitField(pt)

  def newPresentEmitField(name: String, pt: PType): PresentEmitSettable = ecb.newPresentEmitField(name, pt)

  def newPresentEmitSettable(pt: PType, ps: PSettable): PresentEmitSettable = ecb.newPresentEmitSettable(pt, ps)

  def fieldBuilder: SettableBuilder = cb.fieldBuilder

  def result(print: Option[PrintWriter] = None): () => C = cb.result(print)

  def getFS: Code[FS] = ecb.getFS

  def getObject[T <: AnyRef : TypeInfo](obj: T): Code[T] = ecb.getObject(obj)

  def getSerializedAgg(i: Int): Code[Array[Byte]] = ecb.getSerializedAgg(i)

  def setSerializedAgg(i: Int, b: Code[Array[Byte]]): Code[Unit] = ecb.setSerializedAgg(i, b)

  def freeSerializedAgg(i: Int): Code[Unit] = ecb.freeSerializedAgg(i)

  def backend(): Code[BackendUtils] = ecb.backend()

  def addModule(name: String, mod: (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]): Unit =
    ecb.addModule(name, mod)

  def partitionRegion: Settable[Region] = ecb.partitionRegion

  def addLiteral(v: Any, t: PType): PValue = ecb.addLiteral(v, t)

  def addEncodedLiteral(encodedLiteral: EncodedLiteral) = ecb.addEncodedLiteral(encodedLiteral)

  def getPType(t: PType): Code[PType] = ecb.getPType(t)

  def getType(t: Type): Code[Type] = ecb.getType(t)

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

  def genDependentFunction[F](baseName: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_])(implicit fti: TypeInfo[F]): DependentEmitFunctionBuilder[F] =
    ecb.genDependentFunction(baseName, maybeGenericParameterTypeInfo, maybeGenericReturnTypeInfo)

  def newRNG(seed: Long): Value[IRRandomness] = ecb.newRNG(seed)

  def resultWithIndex(print: Option[PrintWriter] = None): (Int, Region) => C = ecb.resultWithIndex(print)

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

  def open(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    getFS.invoke[String, Boolean, InputStream]("open", path, checkCodec)

  def create(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("create", path)

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](
    baseName: String = null
  ): DependentEmitFunctionBuilder[AsmFunction2[A1, A2, R]] =
    genDependentFunction[AsmFunction2[A1, A2, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo]: DependentEmitFunctionBuilder[AsmFunction3[A1, A2, A3, R]] =
    genDependentFunction[AsmFunction3[A1, A2, A3, R]](null, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])
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

  def result(print: Option[PrintWriter] = None): () => C = cb.result(print)

  // EmitClassBuilder methods

  def newPSettable(sb: SettableBuilder, pt: PType, name: String = null): PSettable = PSettable(sb, pt, name)

  def newPField(pt: PType): PSettable = newPSettable(fieldBuilder, pt)

  def newPField(name: String, pt: PType): PSettable = newPSettable(fieldBuilder, pt, name)

  def newEmitField(pt: PType): EmitSettable =
    newEmitSettable(pt, genFieldThisRef[Boolean](), newPField(pt))

  def newEmitField(name: String, pt: PType): EmitSettable =
    newEmitSettable(pt, genFieldThisRef[Boolean](name + "_missing"), newPField(name, pt))

  def newEmitSettable(_pt: PType, ms: Settable[Boolean], vs: PSettable): EmitSettable = new EmitSettable {
    if (!_pt.isRealizable) {
      throw new UnsupportedOperationException(s"newEmitSettable can only be called on realizable PTypes. Called on ${_pt}")
    }

    def pt: PType = _pt

    def load: EmitCode = EmitCode(Code._empty,
      if (_pt.required) false else ms.get,
      vs.get)

    def store(cb: EmitCodeBuilder, ec: EmitCode): Unit = {
      cb.append(ec.setup)

      if (_pt.required) {
        cb.ifx(ec.m, cb._fatal(s"Required EmitSettable cannot be missing ${ _pt }"))
        cb.assign(vs, ec.pv)
      } else {
        cb.ifx(ec.m,
          cb.assign(ms, true),
          {
            cb.assign(ms, false)
            cb.assign(vs, ec.pv)
          })
      }
    }

    def store(cb: EmitCodeBuilder, iec: IEmitCode): Unit =
      if (_pt.required)
        cb.assign(vs, iec.get(cb, s"Required EmitSettable cannot be missing ${ _pt }"))
      else
        iec.consume(cb, {
          cb.assign(ms, true)
        }, { value =>
          cb.assign(ms, false)
          cb.assign(vs, value)
        })

    override def get(cb: EmitCodeBuilder): PValue = {
      if (_pt.required) {
        vs
      } else {
        cb.ifx(ms, cb._fatal(s"Can't convert missing ${_pt} to PValue"))
        vs
      }
    }
  }

  def newPresentEmitField(pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPField(pt))

  def newPresentEmitField(name: String, pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPField(name, pt))

  def newPresentEmitSettable(_pt: PType, ps: PSettable): PresentEmitSettable = new PresentEmitSettable {
    def pt: PType = _pt

    def load: EmitCode = EmitCode(Code._empty, const(false), ps.load())

    def store(cb: EmitCodeBuilder, pv: PCode): Unit = ps.store(cb, pv)

    override def get(cb: EmitCodeBuilder): PValue = ps
  }

  private[this] val typMap: mutable.Map[Type, Value[_ <: Type]] =
    mutable.Map()

  private[this] val pTypeMap: mutable.Map[PType, Value[_ <: PType]] = mutable.Map()

  private[this] type CompareMapKey = (SType, SType)
  private[this] val memoizedComparisons: mutable.Map[CompareMapKey, CodeOrdering] =
    mutable.Map[CompareMapKey, CodeOrdering]()


  def numTypes: Int = typMap.size

  private[this] def addReferenceGenome(rg: ReferenceGenome): Code[Unit] = {
    val rgExists = Code.invokeScalaObject1[String, Boolean](ReferenceGenome.getClass, "hasReference", rg.name)
    val addRG = Code.invokeScalaObject1[ReferenceGenome, Unit](ReferenceGenome.getClass, "addReference", getReferenceGenome(rg))
    rgExists.mux(Code._empty, addRG)
  }

  private[this] val literalsMap: mutable.Map[(PType, Any), PSettable] =
    mutable.Map[(PType, Any), PSettable]()
  private[this] val encodedLiteralsMap: mutable.Map[EncodedLiteral, PSettable] =
    mutable.Map[EncodedLiteral, PSettable]()
  private[this] lazy val encLitField: Settable[Array[Byte]] = genFieldThisRef[Array[Byte]]("encodedLiterals")

  lazy val partitionRegion: Settable[Region] = genFieldThisRef[Region]("partitionRegion")
  private[this] lazy val poolField: Settable[RegionPool] = genFieldThisRef[RegionPool]()

  def addLiteral(v: Any, t: PType): PValue = {
    assert(v != null)
    assert(t.isCanonical)
    literalsMap.getOrElseUpdate(t -> v, PSettable(fieldBuilder, t, "literal"))
  }

  def addEncodedLiteral(encodedLiteral: EncodedLiteral): PValue = {
    assert(encodedLiteral._pType.isCanonical)
    encodedLiteralsMap.getOrElseUpdate(encodedLiteral, PSettable(fieldBuilder, encodedLiteral._pType, "encodedLiteral"))
  }

  private[this] def encodeLiterals(): Array[Array[Byte]] = {
    val literals = literalsMap.toArray
    val litType = PCanonicalTuple(true, literals.map(_._1._1): _*)
    val spec = TypedCodecSpec(litType, BufferSpec.defaultUncompressed)

    cb.addInterface(typeInfo[FunctionWithLiterals].iname)
    val mb2 = newEmitMethod("addLiterals", FastIndexedSeq[ParamType](typeInfo[Array[Array[Byte]]]), typeInfo[Unit])

    val preEncodedLiterals = encodedLiteralsMap.toArray

    mb2.voidWithBuilder { cb =>
      val allEncodedFields = mb2.getCodeParam[Array[Array[Byte]]](1)

      val ib = cb.newLocal[InputBuffer]("ib",
        spec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](allEncodedFields(0))))

      val lits = spec.encodedType.buildDecoder(spec.encodedVirtualType, this)
        .apply(cb, partitionRegion, ib)
        .asBaseStruct
        .memoize(cb, "cb_lits")
      literals.zipWithIndex.foreach { case (((_, _), f), i) =>
        lits.loadField(cb, i)
          .consume(cb,
            cb._fatal("expect non-missing literals!"),
            { pc => f.store(cb, pc.asPCode) })
      }
      // Handle the pre-encoded literals, which only need to be decoded.
      preEncodedLiterals.zipWithIndex.foreach { case ((encLit, f), index) =>
        val spec = encLit.codec
        cb.assign(ib, spec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](allEncodedFields(index + 1))))
        val decodedValue = encLit.codec.encodedType.buildDecoder(encLit.typ, this)
          .apply(cb, partitionRegion, ib)
        assert(decodedValue.st == f.st)

        // Because 0th index is for the regular literals
        f.store(cb, decodedValue)
      }
    }

    val baos = new ByteArrayOutputStream()
    val enc = spec.buildEncoder(ctx, litType)(baos)
    this.emodb.ctx.r.pool.scopedRegion { region =>
      val rvb = new RegionValueBuilder(region)
      rvb.start(litType)
      rvb.startTuple()
      literals.foreach { case ((typ, a), _) => rvb.addAnnotation(typ.virtualType, a) }
      rvb.endTuple()
      enc.writeRegionValue(rvb.end())
    }
    enc.flush()
    enc.close()
    Array(baos.toByteArray) ++ preEncodedLiterals.map(_._1.value.ba)
  }

  private[this] var _objectsField: Settable[Array[AnyRef]] = _
  private[this] var _objects: BoxedArrayBuilder[AnyRef] = _

  private[this] var _mods: BoxedArrayBuilder[(String, (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])] = new BoxedArrayBuilder()
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

    val newF = newEmitMethod("newAggState", FastIndexedSeq[ParamType](typeInfo[Region]), typeInfo[Unit])
    val setF = newEmitMethod("setAggState", FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[Long]), typeInfo[Unit])
    val getF = newEmitMethod("getAggOffset", FastIndexedSeq[ParamType](), typeInfo[Long])
    val storeF = newEmitMethod("storeAggsToRegion", FastIndexedSeq[ParamType](), typeInfo[Unit])
    val setNSer = newEmitMethod("setNumSerialized", FastIndexedSeq[ParamType](typeInfo[Int]), typeInfo[Unit])
    val setSer = newEmitMethod("setSerializedAgg", FastIndexedSeq[ParamType](typeInfo[Int], typeInfo[Array[Byte]]), typeInfo[Unit])
    val getSer = newEmitMethod("getSerializedAgg", FastIndexedSeq[ParamType](typeInfo[Int]), typeInfo[Array[Byte]])

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
      cb += storeF.invokeCode[Unit]()
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
    _aggSerialized.load().update(i, Code._null)
  }

  def backend(): Code[BackendUtils] = {
    if (_backendField == null) {
      cb.addInterface(typeInfo[FunctionWithBackend].iname)
      val backendField = genFieldThisRef[BackendUtils]()
      val mb = newEmitMethod("setBackend", FastIndexedSeq[ParamType](typeInfo[BackendUtils]), typeInfo[Unit])
      mb.emit(backendField := mb.getCodeParam[BackendUtils](1))
      _backendField = backendField
    }
    _backendField
  }

  def pool(): Value[RegionPool] = {
    poolField
  }

  def addModule(name: String, mod: (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]): Unit = {
    _mods += name -> mod
  }

  def getFS: Code[FS] = emodb.getFS

  def getObject[T <: AnyRef : TypeInfo](obj: T): Code[T] = {
    if (_objectsField == null) {
      cb.addInterface(typeInfo[FunctionWithObjects].iname)
      _objectsField = genFieldThisRef[Array[AnyRef]]()
      _objects = new BoxedArrayBuilder[AnyRef]()
      val mb = newEmitMethod("setObjects", FastIndexedSeq[ParamType](typeInfo[Array[AnyRef]]), typeInfo[Unit])
      mb.emit(_objectsField := mb.getCodeParam[Array[AnyRef]](1))
    }

    val i = _objects.size
    _objects += obj
    Code.checkcast[T](toCodeArray(_objectsField).apply(i))
  }

  def getPType[T <: PType : TypeInfo](t: T): Code[T] = {
    val references = ReferenceGenome.getReferences(t.virtualType).toArray
    val setup = Code(Code(references.map(addReferenceGenome)),
      Code.checkcast[T](
        Code.invokeScalaObject1[String, PType](
          IRParser.getClass, "parsePType", t.toString)))
    pTypeMap.getOrElseUpdate(t,
      genLazyFieldThisRef[T](setup)).get.asInstanceOf[Code[T]]
  }

  def getType[T <: Type : TypeInfo](t: T): Code[T] = {
    val references = ReferenceGenome.getReferences(t).toArray
    val setup = Code(Code(references.map(addReferenceGenome)),
      Code.checkcast[T](
        Code.invokeScalaObject1[String, Type](
          IRParser.getClass, "parseType", t.parsableString())))
    typMap.getOrElseUpdate(t,
      genLazyFieldThisRef[T](setup)).get.asInstanceOf[Code[T]]
  }

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
    val ord = getOrdering(t1, t2, sortOrder);

    { (cb: EmitCodeBuilder, v1: EmitCode, v2: EmitCode) =>

      val r = op match {
        case CodeOrdering.Compare(missingEqual) => ord.compare(cb, v1, v2, missingEqual)
        case CodeOrdering.Equiv(missingEqual) => ord.equiv(cb, v1, v2, missingEqual)
        case CodeOrdering.Lt(missingEqual) => ord.lt(cb, v1, v2, missingEqual)
        case CodeOrdering.Lteq(missingEqual) => ord.lteq(cb, v1, v2, missingEqual)
        case CodeOrdering.Gt(missingEqual) => ord.gt(cb, v1, v2, missingEqual)
        case CodeOrdering.Gteq(missingEqual) => ord.gteq(cb, v1, v2, missingEqual)
        case CodeOrdering.Neq(missingEqual) => !ord.equiv(cb, v1, v2, missingEqual)
      }
      coerce[op.ReturnType](r)
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

  private def getCodeArgsInfo(argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): (IndexedSeq[TypeInfo[_]], TypeInfo[_]) = {
    val codeArgsInfo = argsInfo.flatMap {
      case CodeParamType(ti) => FastIndexedSeq(ti)
      case EmitParamType(pt) => EmitCode.codeTupleTypes(pt)
      case PCodeParamType(pt) => pt.codeTupleTypes()
    }
    val codeReturnInfo = returnInfo match {
      case CodeParamType(ti) => ti
      case PCodeParamType(pt) => pt.ti
      case EmitParamType(pt) =>
        val ts = EmitCode.codeTupleTypes(pt)
        if (ts.length == 1)
          ts.head
        else {
          val t = modb.tupleClass(ts)
          t.cb.ti
        }
    }

    (codeArgsInfo, codeReturnInfo)
  }

  def newEmitMethod(name: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] = {
    val (codeArgsInfo, codeReturnInfo) = getCodeArgsInfo(argsInfo, returnInfo)

    new EmitMethodBuilder[C](
      argsInfo, returnInfo,
      this,
      cb.newMethod(name, codeArgsInfo, codeReturnInfo))
  }

  def newEmitMethod(name: String, argsInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]): EmitMethodBuilder[C] = {
    new EmitMethodBuilder[C](
      argsInfo.map(ai => CodeParamType(ai.base)), CodeParamType(returnInfo.base),
      this,
      cb.newMethod(name, argsInfo, returnInfo))
  }

  def newStaticEmitMethod(name: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] = {
    val (codeArgsInfo, codeReturnInfo) = getCodeArgsInfo(argsInfo, returnInfo)

    new EmitMethodBuilder[C](
      argsInfo, returnInfo,
      this,
      cb.newStaticMethod(name, codeArgsInfo, codeReturnInfo))
  }

  def genDependentFunction[F](baseName: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_])(implicit fti: TypeInfo[F]): DependentEmitFunctionBuilder[F] = {
    val depCB = emodb.genEmitClass[F](baseName)
    val apply_method = depCB.cb.newMethod("apply", maybeGenericParameterTypeInfo, maybeGenericReturnTypeInfo)
    val dep_apply_method = new DependentMethodBuilder(apply_method)
    val emit_apply_method = new EmitMethodBuilder[F](
      maybeGenericParameterTypeInfo.map(pi => CodeParamType(pi.base)),
      CodeParamType(maybeGenericReturnTypeInfo.base),
      depCB,
      apply_method)
    new DependentEmitFunctionBuilder[F](this, dep_apply_method, emit_apply_method)
  }

  val rngs: BoxedArrayBuilder[(Settable[IRRandomness], Code[IRRandomness])] = new BoxedArrayBuilder()

  def makeAddPartitionRegion(): Unit = {
    cb.addInterface(typeInfo[FunctionWithPartitionRegion].iname)
    val mb = newEmitMethod("addPartitionRegion", FastIndexedSeq[ParamType](typeInfo[Region]), typeInfo[Unit])
    mb.emit(partitionRegion := mb.getCodeParam[Region](1))
    val mb2 = newEmitMethod("setPool", FastIndexedSeq[ParamType](typeInfo[RegionPool]), typeInfo[Unit])
    mb2.emit(poolField := mb2.getCodeParam[RegionPool](1))
  }

  def makeRNGs() {
    cb.addInterface(typeInfo[FunctionWithSeededRandomness].iname)

    val initialized = genFieldThisRef[Boolean]()
    val mb = newEmitMethod("setPartitionIndex", IndexedSeq[ParamType](typeInfo[Int]), typeInfo[Unit])

    val rngFields = rngs.result()
    val initialize = Code(rngFields.map { case (field, initialization) =>
      field := initialization
    })

    val reseed = Code(rngFields.map { case (field, _) =>
      field.invoke[Int, Unit]("reset", mb.getCodeParam[Int](1))
    })

    mb.emit(Code(
      initialized.mux(
        Code._empty,
        Code(initialize, initialized := true)),
      reseed))
  }

  def newRNG(seed: Long): Value[IRRandomness] = {
    val rng = genFieldThisRef[IRRandomness]()
    rngs += rng -> Code.newInstance[IRRandomness, Long](seed)
    rng
  }

  def resultWithIndex(print: Option[PrintWriter] = None): (Int, Region) => C = {
    makeRNGs()
    makeAddPartitionRegion()

    val hasLiterals: Boolean = literalsMap.nonEmpty || encodedLiteralsMap.nonEmpty

    val literalsBc = if (hasLiterals)
      ctx.backend.broadcast(encodeLiterals())
    else
      // if there are no literals, there might not be a HailContext
      null

    val nSerializedAggs = _nSerialized

    val useBackend = _backendField != null
    val backend = if (useBackend) new BackendUtils(_mods.result()) else null

    val objects =
      if (_objects != null)
        _objects.result()
      else
        null

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    val n = cb.className.replace("/", ".")
    val classesBytes = modb.classesBytes(print)

    new ((Int, Region) => C) with java.io.Serializable {
      @transient @volatile private var theClass: Class[_] = null

      def apply(idx: Int, region: Region): C = {
        if (theClass == null) {
          this.synchronized {
            if (theClass == null) {
              classesBytes.load()
              theClass = loadClass(n)
            }
          }
        }
        val f = theClass.newInstance().asInstanceOf[C]
        f.asInstanceOf[FunctionWithPartitionRegion].addPartitionRegion(region)
        f.asInstanceOf[FunctionWithPartitionRegion].setPool(region.pool)
        if (useBackend)
          f.asInstanceOf[FunctionWithBackend].setBackend(backend)
        if (objects != null)
          f.asInstanceOf[FunctionWithObjects].setObjects(objects)
        if (hasLiterals)
          f.asInstanceOf[FunctionWithLiterals].addLiterals(literalsBc.value)
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
    methodMemo.getOrElseUpdate(key, {
      val mb = genEmitMethod(baseName, argsInfo, returnInfo)
      body(mb)
      mb
    })
  }

  def genEmitMethod(baseName: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    newEmitMethod(genName("m", baseName), argsInfo, returnInfo)

  def genEmitMethod[R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[ParamType](), typeInfo[R])

  def genEmitMethod[A: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[ParamType](typeInfo[A]), typeInfo[R])

  def genEmitMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[ParamType](typeInfo[A], typeInfo[B]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[ParamType](typeInfo[A1], typeInfo[A2], typeInfo[A3]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[ParamType](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[ParamType](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4], typeInfo[A5]), typeInfo[R])

  def genStaticEmitMethod(baseName: String, argsInfo: IndexedSeq[ParamType], returnInfo: ParamType): EmitMethodBuilder[C] =
    newStaticEmitMethod(genName("sm", baseName), argsInfo, returnInfo)

  def getUnsafeReader(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    getFS.invoke[String, Boolean, InputStream]("unsafeReader", path, checkCodec)

  def getUnsafeWriter(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("unsafeWriter", path)

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](
    baseName: String = null
  ): DependentEmitFunctionBuilder[AsmFunction2[A1, A2, R]] =
    genDependentFunction[AsmFunction2[A1, A2, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo]: DependentEmitFunctionBuilder[AsmFunction3[A1, A2, A3, R]] =
    genDependentFunction[AsmFunction3[A1, A2, A3, R]](null, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])
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
    EmitFunctionBuilder[AsmFunction0[R]](ctx, baseName, FastIndexedSeq[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

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

trait FunctionWithPartitionRegion {
  def addPartitionRegion(r: Region): Unit
  def setPool(pool: RegionPool): Unit
}

trait FunctionWithLiterals {
  def addLiterals(lit: Array[Array[Byte]]): Unit
}

trait FunctionWithSeededRandomness {
  def setPartitionIndex(idx: Int): Unit
}

trait FunctionWithBackend {
  def setBackend(spark: BackendUtils): Unit
}

class EmitMethodBuilder[C](
  val emitParamTypes: IndexedSeq[ParamType],
  val emitReturnType: ParamType,
  val ecb: EmitClassBuilder[C],
  val mb: MethodBuilder[C]
) extends WrappedEmitClassBuilder[C] {
  // wrapped MethodBuilder methods
  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] = mb.newLocal[T](name)

  def localBuilder: SettableBuilder = mb.localBuilder

  // FIXME needs to code and emit variants
  def emit(body: Code[_]): Unit = mb.emit(body)

  // EmitMethodBuilder methods

  // this, ...
  private val emitParamCodeIndex = emitParamTypes.scanLeft((!mb.isStatic).toInt) {
    case (i, EmitParamType(pt)) =>
      i + pt.nCodes + (if (pt.required) 0 else 1)
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

  def getPCodeParam(emitIndex: Int): PCode = {
    assert(mb.isStatic || emitIndex != 0)
    val static = (!mb.isStatic).toInt
    val _pt = emitParamTypes(emitIndex - static).asInstanceOf[PCodeParamType].pt
    assert(!_pt.isInstanceOf[PStream])

    val ts = _pt.codeTupleTypes()
    val codeIndex = emitParamCodeIndex(emitIndex - static)

    _pt.sType.fromCodes(ts.zipWithIndex.map { case (t, i) =>
      mb.getArg(codeIndex + i)(t).load()
    }).asPCode
  }

  def getEmitParam(emitIndex: Int): EmitValue = {
    assert(mb.isStatic || emitIndex != 0)
    val static = (!mb.isStatic).toInt
    val _pt = emitParamTypes(emitIndex - static).asInstanceOf[EmitParamType].pt
    assert(!_pt.isInstanceOf[PStream])

    val ts = _pt.codeTupleTypes()
    val codeIndex = emitParamCodeIndex(emitIndex - static)

    new EmitValue {
      evSelf =>
      val pt: PType = _pt

      def load: EmitCode = {
        EmitCode(Code._empty,
          if (pt.required)
            const(false)
          else
            mb.getArg[Boolean](codeIndex + ts.length),
          pt.fromCodeTuple(ts.zipWithIndex.map { case (t, i) =>
            mb.getArg(codeIndex + i)(t).get
          }))
      }

      override def get(cb: EmitCodeBuilder): PValue = {
        new PValue {
          override def pt: PType = evSelf.pt

          override def get: PCode = pt.fromCodeTuple(ts.zipWithIndex.map { case (t, i) =>
            mb.getArg(codeIndex + i)(t).get
          })

          override def st: SType = evSelf.pt.sType
        }
      }
    }
  }

  def getStreamEmitParam(cb: EmitCodeBuilder, emitIndex: Int): IEmitCodeGen[Code[StreamArgType]] = {
    assert(emitIndex != 0)

    val pt = emitParamTypes(emitIndex - 1).asInstanceOf[EmitParamType].pt
    val codeIndex = emitParamCodeIndex(emitIndex - 1)

    val Lpresent = CodeLabel()
    val Lmissing = CodeLabel()

    if (pt.required) {
      cb.goto(Lpresent)
    } else {
      cb.ifx(mb.getArg[Boolean](codeIndex + 1), {
        cb.goto(Lmissing)
      }, {
        cb.goto(Lpresent)
      })
    }

    IEmitCodeGen(Lmissing, Lpresent, mb.getArg[StreamArgType](codeIndex))
  }

  def getParamsList(): IndexedSeq[Param] = {
    emitParamTypes.toFastIndexedSeq.zipWithIndex.map {
      case (CodeParamType(ti), i) => CodeParam(this.getCodeParam(i + 1)(ti)): Param
      case (PCodeParamType(pt), i) => PCodeParam(this.getPCodeParam(i + 1)): Param
      case (EmitParamType(pt), i) => EmitParam(this.getEmitParam(i + 1)): Param
    }
  }

  def invokeCode[T](args: Param*): Code[T] = {
    assert(emitReturnType.isInstanceOf[CodeParamType])
    assert(args.forall(_.isInstanceOf[CodeParam]))
    mb.invoke(args.flatMap {
      case CodeParam(c) => FastIndexedSeq(c)
      // If you hit this assertion, it means that an EmitParam was passed to
      // invokeCode. Code with EmitParams must be invoked using the EmitCodeBuilder
      // interface to ensure that setup is run and missingness is evaluated for the
      // EmitCode
      case EmitParam(ec) => fatal("EmitParam passed to invokeCode")
    }: _*)
  }
  def newPLocal(pt: PType): PSettable = newPSettable(localBuilder, pt)

  def newPLocal(name: String, pt: PType): PSettable = newPSettable(localBuilder, pt, name)

  def newEmitLocal(pt: PType): EmitSettable =
    newEmitSettable(pt, if (pt.required) null else newLocal[Boolean](), newPLocal(pt))

  def newEmitLocal(name: String, pt: PType): EmitSettable =
    newEmitSettable(pt, if (pt.required) null else newLocal[Boolean](name + "_missing"), newPLocal(name, pt))

  def newPresentEmitLocal(pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPLocal(pt))

  def newPresentEmitLocal(name: String, pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPLocal(name, pt))

  def emitWithBuilder[T](f: (EmitCodeBuilder) => Code[T]): Unit = emit(EmitCodeBuilder.scopedCode[T](this)(f))

  def voidWithBuilder(f: (EmitCodeBuilder) => Unit): Unit = emit(EmitCodeBuilder.scopedVoid(this)(f))

  def emitPCode(f: (EmitCodeBuilder) => PCode): Unit = {
    // FIXME: this should optionally construct a tuple to support multiple-code SCodes
    emit(EmitCodeBuilder.scopedCode(this) { cb =>
      val res = f(cb)
      res.code
    })
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

  def getEmitParam(emitIndex: Int): EmitValue = emb.getEmitParam(emitIndex)

  def newPLocal(pt: PType): PSettable = emb.newPLocal(pt)

  def newPLocal(name: String, pt: PType): PSettable = emb.newPLocal(name, pt)

  def newEmitLocal(pt: PType): EmitSettable = emb.newEmitLocal(pt)

  def newEmitLocal(name: String, pt: PType): EmitSettable = emb.newEmitLocal(name, pt)

  def newPresentEmitLocal(pt: PType): PresentEmitSettable = emb.newPresentEmitLocal(pt)
}

class DependentEmitFunctionBuilder[F](
  parentcb: EmitClassBuilder[_],
  val dep_apply_method: DependentMethodBuilder[F],
  val apply_method: EmitMethodBuilder[F]
) extends WrappedEmitMethodBuilder[F] {
  def emb: EmitMethodBuilder[F] = apply_method

  // wrapped DependentMethodBuilder
  def newDepField[T : TypeInfo](value: Code[T]): Value[T] = dep_apply_method.newDepField[T](value)

  def newDepFieldAny[T: TypeInfo](value: Code[_]): Value[T] = dep_apply_method.newDepFieldAny[T](value)

  def newInstance(mb: EmitMethodBuilder[_]): Code[F] = dep_apply_method.newInstance(mb.mb)

  private[this] val typMap: mutable.Map[Type, Value[Type]] =
    mutable.Map[Type, Value[Type]]()

  private[this] val literalsMap: mutable.Map[(PType, Any), PValue] =
    mutable.Map[(PType, Any), PValue]()

  override def getType(t: Type): Code[Type] =
    typMap.getOrElseUpdate(t, {
      val fromParent = parentcb.getType(t)
      val field = newDepField[Type](fromParent)
      field
    })

  override def addLiteral(v: Any, t: PType): PValue = {
    assert(v != null)
    literalsMap.getOrElseUpdate(t -> v, {
      val fromParent = parentcb.addLiteral(v, t)
      newDepPField(fromParent.get)
    })
  }

  def newDepPField(pc: PCode): PValue = {
    val ti = typeToTypeInfo(pc.pt)
    val field = newPField(pc.pt)
    dep_apply_method.setFields += { (obj: lir.ValueX) =>
      val code = pc.code
      // XXX below assumes that the first settable is the 'base' of the PSettable
      val baseField = field.settableTuple()(0).asInstanceOf[ThisFieldRef[_]]
      code.end.append(lir.putField(className, baseField.name, ti, obj, code.v))
      // FIXME need to initialize other potential settables in the PSettable here
      val newC = new VCode(code.start, code.end, null)
      code.clear()
      newC
    }
    field
  }

  def newDepEmitField(ec: EmitCode): EmitValue = {
    val _pt = ec.pt
    val ti = typeToTypeInfo(_pt)
    val m = genFieldThisRef[Boolean]()
    val v = genFieldThisRef()(ti)
    dep_apply_method.setFields += { (obj: lir.ValueX) =>
      val setup = ec.setup
      setup.end.append(lir.goto(ec.m.start))
      ec.m.end.append(lir.putField(className, m.name, typeInfo[Boolean], obj, ec.m.v))
      ec.m.end.append(lir.putField(className, v.name, ti, obj, ec.v.v))
      val newC = new VCode(setup.start, ec.m.end, null)
      setup.clear()
      ec.m.clear()
      ec.v.clear()
      newC
    }
    new EmitValue {
      def pt: PType = _pt

      def get(cb: EmitCodeBuilder): PValue= load.toI(cb).get(
        cb,
        "Can't convert missing value to PValue.").memoize(cb, "newDepEmitField_memo")

      def load: EmitCode = EmitCode(Code._empty, m.load(), PCode(_pt, v.load()))
    }
  }
}

class EmitFunctionBuilder[F](val apply_method: EmitMethodBuilder[F]) extends WrappedEmitMethodBuilder[F] {
  def emb: EmitMethodBuilder[F] = apply_method
}
