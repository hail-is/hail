package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder, EmitRegion, defaultValue, typeToTypeInfo}
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.utils._

class TypedKey(typ: PType, cb: EmitClassBuilder[_], region: Value[Region]) extends BTreeKey {
  val inline: Boolean = typ.isPrimitive
  val storageType: PTuple = PTuple(if (inline) typ else PInt64(typ.required), PTuple())
  val compType: PType = typ
  private val kcomp = cb.getCodeOrdering(typ, CodeOrdering.compare, ignoreMissingness = false)

  def isKeyMissing(src: Code[Long]): Code[Boolean] = storageType.isFieldMissing(src, 0)
  def loadKey(src: Code[Long]): Code[_] = Region.loadIRIntermediate(if (inline) typ else PInt64(typ.required))(storageType.fieldOffset(src, 0))

  def isEmpty(off: Code[Long]): Code[Boolean] = storageType.isFieldMissing(off, 1)
  def initializeEmpty(off: Code[Long]): Code[Unit] =
    storageType.setFieldMissing(off, 1)

  def store(dest: Code[Long], m: Code[Boolean], v: Code[_]): Code[Unit] = {
    Code.memoize(dest, "casa_store_dest") { dest =>
      val c = {
        if (typ.isPrimitive)
          Region.storeIRIntermediate(typ)(storageType.fieldOffset(dest, 0), v)
        else
          Region.storeAddress(storageType.fieldOffset(dest, 0), StagedRegionValueBuilder.deepCopyFromOffset(cb, region, typ, coerce[Long](v)))
      }
      if (!typ.required)
        m.mux(
          Code(storageType.setFieldPresent(dest, 1), storageType.setFieldMissing(dest, 0)),
          Code(storageType.stagedInitialize(dest), c))
      else
        Code(storageType.setFieldPresent(dest, 1), c)
    }
  }

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit] =
    Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(er: EmitRegion, dest: Code[Long], src: Code[Long]): Code[Unit] = {
    if (inline)
      StagedRegionValueBuilder.deepCopy(er, storageType, src, dest)
    else
      Region.storeAddress(dest, StagedRegionValueBuilder.deepCopyFromOffset(er, storageType, src))
  }

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] =
    kcomp(k1, k2)

  def loadCompKey(off: Value[Long]): (Code[Boolean], Code[_]) = {
    isKeyMissing(off) -> isKeyMissing(off).mux(defaultValue(typ), loadKey(off))
  }
}

class AppendOnlySetState(val cb: EmitClassBuilder[_], t: PType) extends PointerBasedRVAState {
  val root: Settable[Long] = cb.genFieldThisRef[Long]()
  val size: Settable[Int] = cb.genFieldThisRef[Int]()
  val key = new TypedKey(t, cb, region)
  val tree = new AppendOnlyBTree(cb, key, region, root)
  val et = EType.defaultFromPType(t)

  val typ: PStruct = PStruct(
    required = true,
    "size" -> PInt32(true),
    "tree" -> PInt64(true))

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    Code(super.load(regionLoader, src),
      off.ceq(0L).mux(Code._empty,
        Code(
          size := Region.loadInt(typ.loadField(off, 0)),
          root := Region.loadAddress(typ.loadField(off, 1)))))
  }

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] = {
    Code(
      Region.storeInt(typ.fieldOffset(off, 0), size),
      Region.storeAddress(typ.fieldOffset(off, 1), root),
      super.store(regionStorer, dest))
  }

  def init: Code[Unit] = Code(
    off := region.allocate(typ.alignment, typ.byteSize),
    size := 0, tree.init)

  private val _elt = cb.genFieldThisRef[Long]()
  private val _vm = cb.genFieldThisRef[Boolean]()
  private val _vv = cb.genFieldThisRef()(typeToTypeInfo(t))
  def insert(vm: Code[Boolean], vv: Code[_]): Code[Unit] = {
    Code(
      vm.mux(
        Code(_vm := true, _vv.storeAny(defaultValue(t))),
        Code(_vm := false, _vv.storeAny(vv))),
      _elt := tree.getOrElseInitialize(_vm, _vv),
      key.isEmpty(_elt).orEmpty(Code(
        size := size + 1,
        key.store(_elt, _vm, _vv)
      )))
  }

  // loads container; does not update.
  def foreach(f: (Code[Boolean], Code[_]) => Code[Unit]): Code[Unit] =
    tree.foreach { eoff =>
      Code.memoize(eoff, "casa_foreach_eoff") { eoff =>
        f(key.isKeyMissing(eoff), key.loadKey(eoff))
      }
    }

  def copyFromAddress(src: Code[Long]): Code[Unit] =
    Code.memoize(src, "aoss_copy_from_addr_src") { src =>
      Code(
        off := region.allocate(typ.alignment, typ.byteSize),
        size := Region.loadInt(typ.loadField(src, 0)),
        tree.init,
        tree.deepCopy(Region.loadAddress(typ.loadField(src, 1))))
    }

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    val kEnc = et.buildEncoderMethod(t, cb)

    { ob: Value[OutputBuffer] =>
      tree.bulkStore(ob) { (ob, src) =>
        Code.memoize(src, "aoss_ser_src") { src =>
          Code(
            ob.writeBoolean(key.isKeyMissing(src)),
            (!key.isKeyMissing(src)).orEmpty(
              kEnc.invoke(key.loadKey(src), ob)))
        }
      }
    }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    val kDec = et.buildDecoderMethod(t, cb)
    val km = cb.genFieldThisRef[Boolean]("km")
    val kv = cb.genFieldThisRef("kv")(typeToTypeInfo(t))

    { ib: Value[InputBuffer] =>
      Code(
        init,
        tree.bulkLoad(ib) { (ib, dest) =>
          Code(
            km := ib.readBoolean(),
            (!km).orEmpty(kv.storeAny(kDec.invoke(region, ib))),
            key.store(dest, km, kv),
            size := size + 1)
        })
    }
  }
}

class CollectAsSetAggregator(t: PType) extends StagedAggregator {
  type State = AppendOnlySetState

  assert(t.isCanonical)
  val resultType: PSet = PCanonicalSet(t)

  def createState(cb: EmitClassBuilder[_]): State = new AppendOnlySetState(cb, t)

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    state.init
  }

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(elt) = seq
    Code(elt.setup, state.insert(elt.m, elt.v))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] =
    other.foreach { (km, kv) => state.insert(km, kv) }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addArray(resultType.arrayFundamentalType, sab =>
      Code(
        sab.start(state.size),
        state.foreach { (km, kv) =>
          Code(
            km.mux(
              sab.setMissing(),
              sab.addWithDeepCopy(t, coerce[Long](kv))),
            sab.advance())
        }))
}
