package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder, EmitMethodBuilder, EmitRegion, defaultValue, typeToTypeInfo}
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.utils._

class GroupedBTreeKey(kt: PType, cb: EmitClassBuilder[_], region: Value[Region], val offset: Value[Long], states: StateTuple) extends BTreeKey {
  val storageType: PStruct = PCanonicalStruct(required = true,
    "kt" -> kt,
    "regionIdx" -> PInt32(true),
    "container" -> states.storageType)
  val compType: PType = kt
  private val kcomp = cb.getCodeOrdering(kt, CodeOrdering.compare, ignoreMissingness = false)

  private val compLoader: EmitMethodBuilder[_] = {
    val mb = cb.genEmitMethod("compWithKey", Array[TypeInfo[_]](typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(compType)), typeInfo[Int])
    val off = mb.getArg[Long](1)
    val m = mb.getArg[Boolean](2)
    val v = mb.getArg(3)(typeToTypeInfo(compType))
    mb.emit(compKeys(isKeyMissing(off) -> loadKey(off), m.get -> v.get))
    mb
  }

  override def compWithKey(off: Code[Long], k: (Code[Boolean], Code[_])): Code[Int] =
    compLoader.invoke[Int](off, k._1, k._2)

  val regionIdx: Value[Int] = new Value[Int] {
    def get: Code[Int] = Region.loadInt(storageType.fieldOffset(offset, 1))
  }
  val container = new TupleAggregatorState(cb, states, region, containerOffset(offset), regionIdx)

  def isKeyMissing(off: Code[Long]): Code[Boolean] =
    storageType.isFieldMissing(off, 0)
  def loadKey(off: Code[Long]): Code[_] = Region.loadIRIntermediate(kt)(storageType.fieldOffset(off, 0))

  def initValue(dest: Code[Long], km: Code[Boolean], kv: Code[_], rIdx: Code[Int]): Code[Unit] = {
    Code.memoize(dest, "ga_init_value_dest") { dest =>
      val koff = storageType.fieldOffset(dest, 0)
      var storeK =
        if (kt.isPrimitive)
          Region.storeIRIntermediate(kt)(koff, kv)
        else
          StagedRegionValueBuilder.deepCopy(cb, region, kt, coerce[Long](kv), koff)
      if (!kt.required)
        storeK = km.mux(storageType.setFieldMissing(dest, 0), Code(storageType.setFieldPresent(dest, 0), storeK))

      Code(
        storeK,
        storeRegionIdx(dest, rIdx),
        container.newState)
    }
  }

  def loadStates: Code[Unit] = container.load
  def storeStates: Code[Unit] = container.store
  def copyStatesFrom(srcOff: Code[Long]): Code[Unit] = container.copyFrom(srcOff)

  def storeRegionIdx(off: Code[Long], idx: Code[Int]): Code[Unit] =
    Region.storeInt(storageType.fieldOffset(off, 1), idx)

  def containerOffset(off: Value[Long]): Value[Long] = new Value[Long] {
    def get: Code[Long] = storageType.fieldOffset(off, 2)
  }

  def isEmpty(off: Code[Long]): Code[Boolean] =
    Region.loadInt(storageType.fieldOffset(off, 1)) < 0
  def initializeEmpty(off: Code[Long]): Code[Unit] =
    Region.storeInt(storageType.fieldOffset(off, 1), -1)

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit] =
    Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(er: EmitRegion, dest: Code[Long], src: Code[Long]): Code[Unit] =
    Code.memoize(src, "ga_deep_copy_src") { src =>
      Code(StagedRegionValueBuilder.deepCopy(er, storageType, src, dest),
        container.copyFrom(containerOffset(src)),
        container.store)
    }

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] =
    kcomp(k1, k2)

  def loadCompKey(off: Value[Long]): (Code[Boolean], Code[_]) =
    isKeyMissing(off) -> isKeyMissing(off).mux(defaultValue(kt), loadKey(off))

}

class DictState(val cb: EmitClassBuilder[_], val keyType: PType, val nested: StateTuple) extends PointerBasedRVAState {
  val nStates: Int = nested.nStates
  val valueType: PStruct = PStruct("regionIdx" -> PInt32(true), "states" -> nested.storageType)
  val root: Settable[Long] = cb.genFieldThisRef[Long]()
  val size: Settable[Int] = cb.genFieldThisRef[Int]()
  val keyEType = EType.defaultFromPType(keyType)

  val typ: PStruct = PStruct(
    required = true,
    "inits" -> nested.storageType,
    "size" -> PInt32(true),
    "tree" -> PInt64(true))

  private val _elt = cb.genFieldThisRef[Long]()
  private val initStatesOffset: Value[Long] = new Value[Long] {
    def get: Code[Long] = typ.loadField(off, 0)
  }
  val initContainer: TupleAggregatorState = new TupleAggregatorState(cb, nested, region, initStatesOffset)

  val keyed = new GroupedBTreeKey(keyType, cb, region, _elt, nested)
  val tree = new AppendOnlyBTree(cb, keyed, region, root, maxElements = 6)

  def initElement(eltOff: Code[Long], km: Code[Boolean], kv: Code[_]): Code[Unit] = {
    Code(
      size := size + 1,
      region.setNumParents((size + 1) * nStates),
      keyed.initValue(_elt, km, kv, size * nStates))
  }

  def loadContainer(km: Code[Boolean], kv: Code[_]): Code[Unit] =
    Code.memoize(km, "ga_load_cont_km") { km =>
      Code.memoizeAny(km.mux(defaultValue(keyType), kv), "ga_load_cont_kv") { kv =>
        Code(
          _elt := tree.getOrElseInitialize(km, kv),
          keyed.isEmpty(_elt).mux(Code(
            initElement(_elt, km, kv),
            keyed.copyStatesFrom(initStatesOffset)),
            keyed.loadStates))
      }(typeToTypeInfo(keyType))
    }

  def withContainer(km: Code[Boolean], kv: Code[_], seqOps: Code[Unit]): Code[Unit] =
    Code(loadContainer(km, kv), seqOps, keyed.storeStates)

  override def createState: Code[Unit] = Code(super.createState, nested.createStates(cb))

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    Code(super.load(regionLoader, src),
      off.ceq(0L).mux(Code._empty,
        Code(
          size := Region.loadInt(typ.loadField(off, 1)),
          root := Region.loadAddress(typ.loadField(off, 2)))))
  }

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] = {
    Code(
      Region.storeInt(typ.fieldOffset(off, 1), size),
      Region.storeAddress(typ.fieldOffset(off, 2), root),
      super.store(regionStorer, dest))
  }

  def init(initOps: Code[Unit]): Code[Unit] = Code(
    region.setNumParents(nStates),
    off := region.allocate(typ.alignment, typ.byteSize),
    initContainer.newState,
    initOps,
    initContainer.store,
    size := 0,
    tree.init)

  def combine(other: DictState, comb: => Code[Unit]): Code[Unit] =
    other.foreach { (km, kv) => withContainer(km, kv, comb) }

  // loads container; does not update.
  def foreach(f: (Code[Boolean], Code[_]) => Code[Unit]): Code[Unit] =
    tree.foreach { kvOff =>
      Code(
        _elt := kvOff, keyed.loadStates,
        f(keyed.isKeyMissing(_elt), keyed.loadKey(_elt)))
    }

  def copyFromAddress(src: Code[Long]): Code[Unit] =
    Code.memoize(src, "ga_copy_from_addr_src") { src =>
      Code(
        init(initContainer.copyFrom(typ.loadField(src, 0))),
        size := Region.loadInt(typ.loadField(src, 1)),
        tree.deepCopy(Region.loadAddress(typ.loadField(src, 2))))
    }

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    val serializers = nested.states.map(_.serialize(codec))
    val kEnc = keyEType.buildEncoderMethod(keyType, cb)
    val km = cb.genFieldThisRef[Boolean]()
    val kv = cb.genFieldThisRef()(typeToTypeInfo(keyType))

    { ob: Value[OutputBuffer] =>
      Code(
        initContainer.load,
        nested.toCodeWithArgs(cb, "grouped_nested_serialize_init", Array[TypeInfo[_]](classInfo[OutputBuffer]),
          IndexedSeq(ob),
          { (i, _, args) =>
            Code.memoize(coerce[OutputBuffer](args.head), "ga_ser_init_ob") { ob => serializers(i)(ob) }
          }),
        tree.bulkStore(ob) { (ob: Value[OutputBuffer], kvOff: Code[Long]) =>
          Code(
            _elt := kvOff,
            km := keyed.isKeyMissing(_elt),
            kv.storeAny(keyed.loadKey(_elt)),
            ob.writeBoolean(km),
            (!km).orEmpty(kEnc.invoke(kv, ob)),
            keyed.loadStates,
            nested.toCodeWithArgs(cb, "grouped_nested_serialize", Array[TypeInfo[_]](classInfo[OutputBuffer]),
              Array(ob.get),
              { (i, _, args) =>
                Code.memoize(coerce[OutputBuffer](args.head), "ga_ser_init_ob") { ob => serializers(i)(ob) }
              }))
        })
    }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    val deserializers = nested.states.map(_.deserialize(codec))
    val kDec = keyEType.buildDecoderMethod(keyType, cb)
    val km = cb.genFieldThisRef[Boolean]()
    val kv = cb.genFieldThisRef()(typeToTypeInfo(keyType))

    { ib: Value[InputBuffer] =>
      Code(
        init(nested.toCodeWithArgs(cb, "grouped_nested_deserialize_init", Array[TypeInfo[_]](classInfo[InputBuffer]),
          FastIndexedSeq(ib),
          { (i, _, args) =>
            Code.memoize(coerce[InputBuffer](args.head), "ga_deser_init_ib") { ib => deserializers(i)(ib) }
          })),
        tree.bulkLoad(ib) { (ib, koff) =>
          Code(
            _elt := koff,
            km := ib.readBoolean(),
            (!km).orEmpty(kv := kDec.invoke(region, ib)),
            initElement(_elt, km, kv),
            nested.toCodeWithArgs(cb, "grouped_nested_deserialize", Array[TypeInfo[_]](classInfo[InputBuffer]),
              FastIndexedSeq(ib),
              { (i, _, args) =>
                Code.memoize(coerce[InputBuffer](args.head), "ga_deser_ib") { ib => deserializers(i)(ib) }
              }),
            keyed.storeStates)
        })
    }
  }
}

class GroupedAggregator(kt: PType, nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = DictState

  assert(kt.isCanonical)
  val resultEltType: PTuple = PCanonicalTuple(true, nestedAggs.map(_.resultType): _*)
  val resultType: PDict = PCanonicalDict(kt, resultEltType)

  def createState(cb: EmitClassBuilder[_]): State = new DictState(cb, kt, StateTuple(nestedAggs.map(_.createState(cb))))

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(inits) = init
    state.init(inits.setup)
  }

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(key, seqs) = seq
    Code(key.setup, state.withContainer(key.m, key.v, seqs.setup))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    state.combine(other, state.nested.toCode(state.cb, "grouped_nested_comb", (i, s) => nestedAggs(i).combOp(s, other.nested(i))))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addArray(resultType.arrayFundamentalType, sab =>
      Code(
        sab.start(state.size),
        state.foreach { (km, kv) =>
          Code(
            sab.addBaseStruct(resultType.elementType, ssb =>
              Code(
                ssb.start(),
                km.mux(
                  ssb.setMissing(),
                  ssb.addWithDeepCopy(kt, kv)),
                ssb.advance(),
                ssb.addBaseStruct(resultEltType, { svb =>
                  Code(svb.start(),
                    state.nested.toCode(state.cb, "grouped_result", { (i, s) =>
                      Code(nestedAggs(i).result(s, svb), svb.advance())
                    }))
                }))),
            sab.advance())
        }))
}
