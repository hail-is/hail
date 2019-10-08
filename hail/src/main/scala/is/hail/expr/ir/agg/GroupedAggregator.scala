package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion, EmitTriplet, defaultValue, typeToTypeInfo}
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.utils._

class GroupedBTreeKey(kt: PType, fb: EmitFunctionBuilder[_], region: Code[Region], val offset: Code[Long], states: StateTuple) extends BTreeKey {
  val storageType: PStruct = PStruct(required = true,
    "kt" -> kt,
    "regionIdx" -> PInt32(true),
    "container" -> PInt64(true))

  val compType: PType = kt
  private val kcomp = fb.getCodeOrdering[Int](kt, CodeOrdering.compare, ignoreMissingness = false)

  val regionIdx: Code[Int] = Region.loadInt(storageType.fieldOffset(offset, 1))
  val container = new TupleAggregatorState(fb, states, region, containerAddress(offset), regionIdx)

  def isKeyMissing(off: Code[Long]): Code[Boolean] =
    storageType.isFieldMissing(off, 0)
  def loadKey(off: Code[Long]): Code[_] = Region.loadIRIntermediate(kt)(storageType.fieldOffset(off, 0))

  def initValue(dest: Code[Long], km: Code[Boolean], kv: Code[_], rIdx: Code[Int]): Code[Unit] = {
    val koff = storageType.fieldOffset(dest, 0)
    var storeK =
      if (kt.isPrimitive)
        Region.storeIRIntermediate(kt)(koff, kv)
      else
        StagedRegionValueBuilder.deepCopy(fb, region, kt, coerce[Long](kv), koff)
    if (!kt.required)
      storeK = km.mux(storageType.setFieldMissing(dest, 0), Code(storageType.setFieldPresent(dest, 0), storeK))

    Code(
      storeK,
      storeRegionIdx(dest, rIdx),
      Region.storeAddress(containerOffset(dest), region.allocate(states.storageType.alignment, states.storageType.byteSize)),
      container.newState)
  }

  def loadStates: Code[Unit] = container.load
  def storeStates: Code[Unit] = container.store
  def copyStatesFrom(srcOff: Code[Long]): Code[Unit] = container.copyFrom(srcOff)

  def storeRegionIdx(off: Code[Long], idx: Code[Int]): Code[Unit] =
    Region.storeInt(storageType.fieldOffset(off, 1), idx)

  def containerOffset(off: Code[Long]): Code[Long] =
    storageType.fieldOffset(off, 2)

  def containerAddress(off: Code[Long]): Code[Long] =
    Region.loadAddress(containerOffset(off))

  def isEmpty(off: Code[Long]): Code[Boolean] =
    Region.loadAddress(containerOffset(off)).ceq(0L)
  def initializeEmpty(off: Code[Long]): Code[Unit] =
    Region.storeAddress(containerOffset(off), 0L)

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit] =
    Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(er: EmitRegion, dest: Code[Long], src: Code[Long]): Code[Unit] =
    Code(StagedRegionValueBuilder.deepCopy(er, storageType, src, dest),
      container.copyFrom(containerAddress(src)),
      container.store)

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] =
    kcomp(k1, k2)

  def loadCompKey(off: Code[Long]): (Code[Boolean], Code[_]) =
    isKeyMissing(off) -> isKeyMissing(off).mux(defaultValue(kt), loadKey(off))

}

class DictState(val fb: EmitFunctionBuilder[_], val keyType: PType, val nested: StateTuple) extends PointerBasedRVAState {
  val nStates: Int = nested.nStates
  val valueType: PStruct = PStruct("regionIdx" -> PInt32(true), "states" -> nested.storageType)
  val root: ClassFieldRef[Long] = fb.newField[Long]
  val size: ClassFieldRef[Int] = fb.newField[Int]
  val keyEType = EType.defaultFromPType(keyType)

  val typ: PStruct = PStruct(
    required = true,
    "inits" -> nested.storageType,
    "size" -> PInt32(true),
    "tree" -> PInt64(true))

  private val _elt = fb.newField[Long]
  private val initStatesOffset: Code[Long] = typ.loadField(off, 0)
  val initContainer: TupleAggregatorState = new TupleAggregatorState(fb, nested, region, initStatesOffset)

  val keyed = new GroupedBTreeKey(keyType, fb, region, _elt, nested)
  val tree = new AppendOnlyBTree(fb, keyed, region, root)

  def initElement(eltOff: Code[Long], km: Code[Boolean], kv: Code[_]): Code[Unit] = {
    Code(
      size := size + 1,
      region.setNumParents((size + 1) * nStates),
      keyed.initValue(_elt, km, kv, size * nStates))
  }

  def loadContainer(km: Code[Boolean], kv: Code[_]): Code[Unit] =
    Code(
      _elt := tree.getOrElseInitialize(km, km.mux(defaultValue(keyType), kv)),
      keyed.isEmpty(_elt).mux(Code(
        initElement(_elt, km, kv),
        keyed.copyStatesFrom(initStatesOffset)),
        keyed.loadStates))

  def withContainer(km: Code[Boolean], kv: Code[_], seqOps: Code[Unit]): Code[Unit] =
    Code(loadContainer(km, kv), seqOps, keyed.storeStates)

  override def createState: Code[Unit] = Code(super.createState, nested.createStates(fb))

  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    Code(super.load(regionLoader, src),
      off.ceq(0L).mux(Code._empty,
        Code(
          size := Region.loadInt(typ.loadField(off, 1)),
          root := Region.loadAddress(typ.loadField(off, 2)))))
  }

  override def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] = {
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

  def combine(other: DictState, comb: Code[Unit]): Code[Unit] =
    other.foreach { (km, kv) => withContainer(km, kv, comb) }

  // loads container; does not update.
  def foreach(f: (Code[Boolean], Code[_]) => Code[Unit]): Code[Unit] =
    tree.foreach { kvOff =>
      Code(
        _elt := kvOff, keyed.loadStates,
        f(keyed.isKeyMissing(_elt), keyed.loadKey(_elt)))
    }

  def copyFromAddress(src: Code[Long]): Code[Unit] =
    Code(
      init(initContainer.copyFrom(typ.loadField(src, 0))),
      size := Region.loadInt(typ.loadField(src, 1)),
      tree.deepCopy(Region.loadAddress(typ.loadField(src, 2))))

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    val serializers = nested.states.map(_.serialize(codec))
    val kEnc = keyEType.buildEncoderMethod(keyType, fb)
    val km = fb.newField[Boolean]
    val kv = fb.newField()(typeToTypeInfo(keyType))

    { ob: Code[OutputBuffer] =>
      Code(
        initContainer.load,
        nested.toCodeWithArgs(fb, "grouped_nested_serialize_init", Array[TypeInfo[_]](classInfo[OutputBuffer]),
          Array(ob),
          { case (i, _, Seq(ob: Code[OutputBuffer@unchecked])) => serializers(i)(ob) }),
        tree.bulkStore(ob) { (ob: Code[OutputBuffer], kvOff: Code[Long]) =>
          Code(
            _elt := kvOff,
            km := keyed.isKeyMissing(_elt),
            kv.storeAny(keyed.loadKey(_elt)),
            ob.writeBoolean(km),
            (!km).orEmpty(kEnc.invoke(kv, ob)),
            keyed.loadStates,
            nested.toCodeWithArgs(fb, "grouped_nested_serialize", Array[TypeInfo[_]](classInfo[OutputBuffer]),
              Array(ob),
              { case (i, _, Seq(ob: Code[OutputBuffer@unchecked])) => serializers(i)(ob) })
          )
        })
    }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val deserializers = nested.states.map(_.deserialize(codec))
    val kDec = keyEType.buildDecoderMethod(keyType, fb)
    val km = fb.newField[Boolean]
    val kv = fb.newField()(typeToTypeInfo(keyType))

    { ib: Code[InputBuffer] =>
      Code(
        init(nested.toCodeWithArgs(fb, "grouped_nested_deserialize_init", Array[TypeInfo[_]](classInfo[InputBuffer]),
          Array(ib),
          { case (i, _, Seq(ib: Code[InputBuffer@unchecked])) => deserializers(i)(ib) })),
        tree.bulkLoad(ib) { (ib, koff) =>
          Code(
            _elt := koff,
            km := ib.readBoolean(),
            (!km).orEmpty(kv := kDec.invoke(region, ib)),
            initElement(_elt, km, kv),
            nested.toCodeWithArgs(fb, "grouped_nested_deserialize", Array[TypeInfo[_]](classInfo[InputBuffer]),
              Array(ib),
              { case (i, _, Seq(ib: Code[InputBuffer@unchecked])) => deserializers(i)(ib) }),
            keyed.storeStates)
        })
    }
  }
}

class GroupedAggregator(kt: PType, nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = DictState

  val resultEltType: PTuple = PTuple(nestedAggs.map(_.resultType): _*)
  val resultType: PDict = PDict(kt, resultEltType)

  def createState(fb: EmitFunctionBuilder[_]): State = new DictState(fb, kt, StateTuple(nestedAggs.map(_.createState(fb))))

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(inits) = init
    state.init(inits.setup)
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(key, seqs) = seq
    Code(key.setup, state.withContainer(key.m, key.v, seqs.setup))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    state.combine(other, state.nested.toCode(state.fb, "grouped_nested_comb", (i, s) => nestedAggs(i).combOp(s, other.nested(i))))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addArray(resultType.fundamentalType.asPArray, sab =>
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
                    state.nested.toCode(state.fb, "grouped_result", { (i, s) =>
                      Code(nestedAggs(i).result(s, svb), svb.advance())
                    }))
                }))),
            sab.advance())
        }))
}
