package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion, EmitTriplet, defaultValue, typeToTypeInfo}
import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.utils._

class GroupedBTreeKey(kt: PType, fb: EmitFunctionBuilder[_], region: Code[Region], container: StateContainer) extends BTreeKey {
  val storageType: PStruct = PStruct(required = true,
    "kt" -> kt,
    "regionIdx" -> PInt32(true),
    "container" -> PInt64(true))

  val compType: PType = kt
  private val kcomp = fb.getCodeOrdering[Int](kt, CodeOrdering.compare, ignoreMissingness = false)

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
      Region.storeAddress(containerOffset(dest), region.allocate(container.typ.alignment, container.typ.byteSize)))
  }

  def regionIdx(off: Code[Long]): Code[Int] =
    Region.loadInt(storageType.fieldOffset(off, 1))

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
      container.toCode((i, s) => s.copyFrom(container.getStateOffset(containerAddress(src), i))),
      container.store(regionIdx(dest), containerAddress(dest)))

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] =
    kcomp(k1, k2)

  def loadCompKey(off: Code[Long]): (Code[Boolean], Code[_]) =
    isKeyMissing(off) -> isKeyMissing(off).mux(defaultValue(kt), loadKey(off))

}

class DictState(val fb: EmitFunctionBuilder[_], val keyType: PType, val nested: Array[AggregatorState]) extends PointerBasedRVAState {
  val nStates: Int = nested.length
  val container: StateContainer = new StateContainer(nested, region)
  val valueType: PStruct = PStruct("regionIdx" -> PInt32(true), "states" -> container.typ)
  val root: ClassFieldRef[Long] = fb.newField[Long]
  val size: ClassFieldRef[Int] = fb.newField[Int]
  val keyed = new GroupedBTreeKey(keyType, fb, region, container)
  val tree = new AppendOnlyBTree(fb, keyed, region, root)

  val typ: PStruct = PStruct(
    required = true,
    "inits" -> container.typ,
    "size" -> PInt32(true),
    "tree" -> PInt64(true))

  private val initStatesOffset: Code[Long] = typ.fieldOffset(off, 0)
  private def initStateOffset(idx: Int): Code[Long] = container.getStateOffset(initStatesOffset, idx)

  private val _elt = fb.newField[Long]

  def initElement(eltOff: Code[Long], km: Code[Boolean], kv: Code[_]): Code[Unit] = {
    Code(
      size := size + 1,
      region.setNumParents(size * (nStates + 1)),
      keyed.initValue(_elt, km, kv, size * nStates),
      container.newStates)
  }

  def loadContainer(km: Code[Boolean], kv: Code[_]): Code[Unit] =
    Code(
      _elt := tree.getOrElseInitialize(km, km.mux(defaultValue(keyType), kv)),
      keyed.isEmpty(_elt).mux(Code(
        initElement(_elt, km, kv),
        container.toCode((i, s) => s.copyFrom(initStateOffset(i)))),
        container.load(keyed.regionIdx(_elt), keyed.containerAddress(_elt))))

  def withContainer(km: Code[Boolean], kv: Code[_], seqOps: Code[Unit]): Code[Unit] =
    Code(loadContainer(km, kv), seqOps, container.store(keyed.regionIdx(_elt), keyed.containerAddress(_elt)))

  override def createState: Code[Unit] = Code(
    super.createState,
    container.toCode((_, s) => s.createState))

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
    container.newStates,
    initOps,
    container.store(0, initStatesOffset),
    size := 0,
    tree.init)

  def combine(other: DictState, comb: Code[Unit]): Code[Unit] =
    other.foreach { (km, kv) => withContainer(km, kv, comb) }

  // loads container; does not update.
  def foreach(f: (Code[Boolean], Code[_]) => Code[Unit]): Code[Unit] =
    tree.foreach { kvOff =>
      Code(
        container.load(
          keyed.regionIdx(kvOff),
          keyed.containerAddress(kvOff)),
        f(keyed.isKeyMissing(kvOff), keyed.loadKey(kvOff)))
    }

  def copyFromAddress(src: Code[Long]): Code[Unit] =
    Code(
      init(container.toCode((i, s) => s.copyFrom(container.getStateOffset(typ.loadField(src, 0), i)))),
      size := Region.loadInt(typ.loadField(src, 1)),
      tree.deepCopy(Region.loadAddress(typ.loadField(src, 2))))

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    val serializers = nested.map(_.serialize(codec))
    val kEnc = EmitPackEncoder.buildMethod(keyType, keyType, fb)
    val km = fb.newField[Boolean]
    val kv = fb.newField()(typeToTypeInfo(keyType))

    { ob: Code[OutputBuffer] =>
      Code(
        container.load(0, initStatesOffset),
        container.toCode((i, _) => serializers(i)(ob)),
        tree.bulkStore(ob) { (ob: Code[OutputBuffer], kvOff: Code[Long]) =>
          Code(
            km := keyed.isKeyMissing(kvOff),
            kv.storeAny(keyed.loadKey(kvOff)),
            ob.writeBoolean(km),
            (!km).orEmpty(kEnc.invoke(region, kv, ob)),
            container.load(
              keyed.regionIdx(kvOff),
              keyed.containerAddress(kvOff)),
            container.toCode((i, _) => serializers(i)(ob)))
        })
    }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val deserializers = nested.map(_.deserialize(codec))
    val kDec = EmitPackDecoder.buildMethod(keyType, keyType, fb)
    val km = fb.newField[Boolean]
    val kv = fb.newField()(typeToTypeInfo(keyType))

    { ib: Code[InputBuffer] =>
      Code(
        init(container.toCode((i, _) => deserializers(i)(ib))),
        tree.bulkLoad(ib) { (ib, koff) =>
          Code(
            _elt := koff,
            km := ib.readBoolean(),
            (!km).orEmpty(kv := kDec.invoke(region, ib)),
            initElement(_elt, km, kv),
            container.toCode((i, _) => deserializers(i)(ib)),
            container.store(keyed.regionIdx(_elt), keyed.containerAddress(_elt)))
        })
    }
  }
}

class GroupedAggregator(kt: PType, nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = DictState

  val resultEltType: PTuple = PTuple(nestedAggs.map(_.resultType): _*)
  val resultType: PDict = PDict(kt, resultEltType)

  def createState(fb: EmitFunctionBuilder[_]): State = new DictState(fb, kt, nestedAggs.map(_.createState(fb)))

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(inits) = init
    state.init(inits.setup)
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(key, seqs) = seq
    Code(key.setup, state.withContainer(key.m, key.v, seqs.setup))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    state.combine(other, state.container.toCode((i, s) => nestedAggs(i).combOp(s, other.nested(i))))
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
                    state.container.toCode { (i, s) =>
                      Code(nestedAggs(i).result(s, svb), svb.advance())
                    })
                }))),
            sab.advance())
        }))
}