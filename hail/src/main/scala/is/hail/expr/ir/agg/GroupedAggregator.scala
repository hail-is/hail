package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{
  EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitRegion, EmitValue, IEmitCode, ParamType,
}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.io._
import is.hail.types.VirtualTypeWithReq
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.virtual.{TVoid, Type}
import is.hail.utils._

class GroupedBTreeKey(
  kt: PType,
  kb: EmitClassBuilder[_],
  region: Value[Region],
  val offset: Value[Long],
  states: StateTuple,
) extends BTreeKey {
  override val storageType: PStruct = PCanonicalStruct(
    required = true,
    "kt" -> kt,
    "regionIdx" -> PInt32(true),
    "container" -> states.storageType,
  )

  override val compType: PType = kt

  override def compWithKey(cb: EmitCodeBuilder, off: Value[Long], k: EmitValue): Value[Int] = {
    val mb = kb.getOrGenEmitMethod(
      "compWithKey",
      ("compWithKey_grouped_btree", kt, k.emitType),
      FastSeq[ParamType](typeInfo[Long], k.emitParamType),
      typeInfo[Int],
    ) { mb =>
      val comp = kb.getOrderingFunction(compType.sType, k.st, CodeOrdering.Compare())
      mb.emitWithBuilder { cb =>
        val off = mb.getCodeParam[Long](1)
        val ev1 = cb.memoize(loadCompKey(cb, off))
        val ev2 = mb.getEmitParam(cb, 2) // don't need region
        comp(cb, ev1, ev2)
      }
    }
    cb.invokeCode(mb, cb.this_, off, k)
  }

  val regionIdx: Value[Int] = new Value[Int] {
    def get: Code[Int] = Region.loadInt(storageType.fieldOffset(offset, 1))
  }

  val container = new TupleAggregatorState(kb, states, region, containerOffset(offset), regionIdx)

  def isKeyMissing(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    storageType.isFieldMissing(cb, off, 0)

  def loadKey(cb: EmitCodeBuilder, off: Code[Long]): SValue =
    cb.memoizeField(kt.loadCheapSCode(cb, storageType.loadField(off, 0)), "loadKey")

  def initValue(cb: EmitCodeBuilder, destc: Code[Long], k: EmitCode, rIdx: Code[Int]): Unit = {
    val dest = cb.newLocal("ga_init_value_dest", destc)
    k.toI(cb)
      .consume(
        cb,
        storageType.setFieldMissing(cb, dest, 0),
        { sc =>
          storageType.setFieldPresent(cb, dest, 0)
          storageType.fieldType("kt")
            .storeAtAddress(cb, storageType.fieldOffset(dest, 0), region, sc, deepCopy = true)
        },
      )
    storeRegionIdx(cb, dest, rIdx)
    container.newState(cb)
  }

  def loadStates(cb: EmitCodeBuilder): Unit = container.load(cb)

  def storeStates(cb: EmitCodeBuilder): Unit = container.store(cb)

  def copyStatesFrom(cb: EmitCodeBuilder, srcOff: Value[Long]): Unit =
    container.copyFrom(cb, srcOff)

  def storeRegionIdx(cb: EmitCodeBuilder, off: Code[Long], idx: Code[Int]): Unit =
    cb += Region.storeInt(storageType.fieldOffset(off, 1), idx)

  def containerOffset(off: Value[Long]): Value[Long] = new Value[Long] {
    def get: Code[Long] = storageType.fieldOffset(off, 2)
  }

  override def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    cb.memoize(Region.loadInt(storageType.fieldOffset(off, 1)) < 0)

  override def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit =
    cb += Region.storeInt(storageType.fieldOffset(off, 1), -1)

  override def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit =
    storageType.storeAtAddress(
      cb,
      dest,
      region,
      storageType.loadCheapSCode(cb, src),
      deepCopy = false,
    )

  override def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, dest: Code[Long], srcCode: Code[Long])
    : Unit = {
    val src = cb.newLocal("ga_deep_copy_src", srcCode)
    storageType.storeAtAddress(
      cb,
      dest,
      region,
      storageType.loadCheapSCode(cb, src),
      deepCopy = true,
    )
    container.copyFrom(cb, containerOffset(src))
    container.store(cb)
  }

  override def compKeys(cb: EmitCodeBuilder, k1: EmitValue, k2: EmitValue): Value[Int] =
    kb.getOrderingFunction(k1.st, k2.st, CodeOrdering.Compare())(cb, k1, k2)

  override def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitValue =
    cb.memoize(IEmitCode(cb, isKeyMissing(cb, off), loadKey(cb, off)))
}

class DictState(
  val kb: EmitClassBuilder[_],
  val keyVType: VirtualTypeWithReq,
  val nested: StateTuple,
) extends PointerBasedRVAState {
  private val keyType = keyVType.canonicalPType
  val nStates: Int = nested.nStates

  val valueType: PStruct =
    PCanonicalStruct("regionIdx" -> PInt32(true), "states" -> nested.storageType)

  val root: Settable[Long] = kb.genFieldThisRef[Long]("grouped_agg_root")
  val size: Settable[Int] = kb.genFieldThisRef[Int]("grouped_agg_size")
  val keyEType = EType.defaultFromPType(keyType)

  val typ: PStruct = PCanonicalStruct(
    required = true,
    "inits" -> nested.storageType,
    "size" -> PInt32(true),
    "tree" -> PInt64(true),
  )

  private val _elt = kb.genFieldThisRef[Long]()

  private val initStatesOffset: Value[Long] = new Value[Long] {
    def get: Code[Long] = typ.loadField(off, 0)
  }

  val initContainer: TupleAggregatorState =
    new TupleAggregatorState(kb, nested, region, initStatesOffset)

  val keyed = new GroupedBTreeKey(keyType, kb, region, _elt, nested)
  val tree = new AppendOnlyBTree(kb, keyed, region, root, maxElements = 6)

  def initElement(cb: EmitCodeBuilder, eltOff: Code[Long], k: EmitCode): Unit = {
    cb.assign(size, size + 1)
    cb += region.setNumParents((size + 1) * nStates)
    keyed.initValue(cb, _elt, k, size * nStates)
  }

  def loadNode(cb: EmitCodeBuilder, node: Code[Long]): Unit = {
    cb.assign(_elt, node)
    keyed.loadStates(cb)
  }

  def loadContainer(cb: EmitCodeBuilder, kec: EmitCode): Unit = {
    val kev = cb.memoize(kec, "ga_load_cont_k")
    cb.assign(_elt, tree.getOrElseInitialize(cb, kev))
    cb.if_(
      keyed.isEmpty(cb, _elt), {
        initElement(cb, _elt, kev)
        keyed.copyStatesFrom(cb, initStatesOffset)
      },
      keyed.loadStates(cb),
    )
  }

  def withContainer(cb: EmitCodeBuilder, k: EmitCode, seqOps: EmitCodeBuilder => Unit): Unit = {
    loadContainer(cb, k)
    seqOps(cb)
    keyed.storeStates(cb)
  }

  override def createState(cb: EmitCodeBuilder): Unit = {
    super.createState(cb)
    nested.createStates(cb)
  }

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = {
    super.load(cb, regionLoader, src)
    cb.if_(
      off.cne(0L), {
        cb.assign(size, Region.loadInt(typ.loadField(off, 1)))
        cb.assign(root, Region.loadAddress(typ.loadField(off, 2)))
      },
    )
  }

  override def store(
    cb: EmitCodeBuilder,
    regionStorer: (EmitCodeBuilder, Value[Region]) => Unit,
    dest: Value[Long],
  ): Unit = {
    cb += Region.storeInt(typ.fieldOffset(off, 1), size)
    cb += Region.storeAddress(typ.fieldOffset(off, 2), root)
    super.store(cb, regionStorer, dest)
  }

  def init(cb: EmitCodeBuilder, initOps: EmitCodeBuilder => Unit): Unit = {
    cb += region.setNumParents(nStates)
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    initContainer.newState(cb)
    initOps(cb)
    initContainer.store(cb)
    cb.assign(size, 0)
    tree.init(cb)
  }

  def combine(cb: EmitCodeBuilder, other: DictState, comb: EmitCodeBuilder => Unit): Unit =
    other.foreach(cb)((cb, k) => withContainer(cb, k, comb))

  // loads container; does not update.
  def foreach(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, EmitCode) => Unit): Unit =
    tree.foreach(cb) { (cb, kvOff) =>
      cb.assign(_elt, kvOff)
      keyed.loadStates(cb)
      f(
        cb,
        EmitCode.fromI(cb.emb)(cb =>
          IEmitCode(cb, keyed.isKeyMissing(cb, _elt), keyed.loadKey(cb, _elt))
        ),
      )
    }

  def copyFromAddress(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    init(cb, cb => initContainer.copyFrom(cb, cb.memoize(typ.loadField(src, 0))))
    cb.assign(size, Region.loadInt(typ.loadField(src, 1)))
    tree.deepCopy(cb, cb.memoize(Region.loadAddress(typ.loadField(src, 2))))
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val serializers = nested.states.map(_.serialize(codec))

    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      initContainer.load(cb)
      nested.toCodeWithArgs(cb, (cb, i, _) => serializers(i)(cb, ob))
      tree.bulkStore(cb, ob) { (cb: EmitCodeBuilder, ob: Value[OutputBuffer], kvOff: Code[Long]) =>
        cb.assign(_elt, kvOff)
        val km = keyed.isKeyMissing(cb, _elt)
        cb += (ob.writeBoolean(km))
        cb.if_(
          !km, {
            val k = keyed.loadKey(cb, _elt)
            keyEType.buildEncoder(k.st, kb)
              .apply(cb, k, ob)
          },
        )
        keyed.loadStates(cb)
        nested.toCodeWithArgs(cb, (cb, i, _) => serializers(i)(cb, ob))
      }
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val deserializers = nested.states.map(_.deserialize(codec))

    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      init(cb, cb => nested.toCodeWithArgs(cb, (cb, i, _) => deserializers(i)(cb, ib)))
      tree.bulkLoad(cb, ib) { (cb, ib, koff) =>
        cb.assign(_elt, koff)

        val kc = EmitCode.fromI(cb.emb)(cb =>
          IEmitCode(
            cb,
            ib.readBoolean(),
            keyEType.buildDecoder(keyType.virtualType, kb).apply(cb, region, ib),
          )
        )
        initElement(cb, _elt, kc)
        nested.toCodeWithArgs(cb, (cb, i, _) => deserializers(i)(cb, ib))
        keyed.storeStates(cb)
      }
    }
  }
}

class GroupedAggregator(ktV: VirtualTypeWithReq, nestedAggs: Array[StagedAggregator])
    extends StagedAggregator {
  type State = DictState

  private val kt = ktV.canonicalPType

  val resultEltType: PTuple =
    PCanonicalTuple(true, nestedAggs.map(_.resultEmitType.storageType): _*)

  val resultPType: PCanonicalDict = PCanonicalDict(kt, resultEltType)
  override val resultEmitType = EmitType(SIndexablePointer(resultPType), true)
  private[this] val arrayRep = resultPType.arrayRep
  private[this] val dictElt = arrayRep.elementType.asInstanceOf[PCanonicalStruct]
  val initOpTypes: Seq[Type] = Array(TVoid)
  val seqOpTypes: Seq[Type] = Array(ktV.t, TVoid)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(inits) = init
    state.init(cb, cb => cb += inits.asVoid())
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(key, seqs) = seq
    state.withContainer(cb, key, (cb) => cb += seqs.asVoid())
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    state: DictState,
    other: DictState,
  ): Unit =
    state.combine(
      cb,
      other,
      cb => state.nested.toCode((i, s) => nestedAggs(i).combOp(ctx, cb, s, other.nested(i))),
    )

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode = {
    val len = state.size
    val resultAddr = cb.newLocal[Long]("groupedagg_result_addr", resultPType.allocate(region, len))
    arrayRep.stagedInitialize(cb, resultAddr, len, setMissing = false)
    val i = cb.newLocal[Int]("groupedagg_result_i", 0)

    state.foreach(cb) { (cb, k) =>
      val addrAtI =
        cb.newLocal[Long]("groupedagg_result_addr_at_i", arrayRep.elementOffset(resultAddr, len, i))
      dictElt.stagedInitialize(cb, addrAtI, setMissing = false)
      k.toI(cb).consume(
        cb,
        dictElt.setFieldMissing(cb, addrAtI, "key"),
        sc =>
          dictElt.fieldType("key").storeAtAddress(
            cb,
            dictElt.fieldOffset(addrAtI, "key"),
            region,
            sc,
            deepCopy = true,
          ),
      )

      val valueAddr =
        cb.newLocal[Long]("groupedagg_value_addr", dictElt.fieldOffset(addrAtI, "value"))
      resultEltType.stagedInitialize(cb, valueAddr, setMissing = false)
      state.nested.toCode { case (nestedIdx, nestedState) =>
        val nestedAddr = cb.newLocal[Long](
          s"groupedagg_result_nested_addr_$nestedIdx",
          resultEltType.fieldOffset(valueAddr, nestedIdx),
        )
        val nestedRes = nestedAggs(nestedIdx).result(cb, nestedState, region)
        nestedRes.consume(
          cb,
          resultEltType.setFieldMissing(cb, valueAddr, nestedIdx),
          sv => resultEltType.types(nestedIdx).storeAtAddress(cb, nestedAddr, region, sv, true),
        ) // TODO: Should this be deep copied?
      }

      cb.assign(i, i + 1)
    }

    // don't need to deep copy because that's done in nested aggregators
    IEmitCode.present(cb, resultPType.loadCheapSCode(cb, resultAddr))
  }
}
