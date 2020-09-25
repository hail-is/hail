package is.hail.expr.ir.agg

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitRegion}
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.io._
import is.hail.utils._

class TypedKey(typ: PType, kb: EmitClassBuilder[_], region: Value[Region]) extends BTreeKey {
  val inline: Boolean = typ.isPrimitive
  val storageType: PTuple = PCanonicalTuple(false, if (inline) typ else PInt64(typ.required), PCanonicalTuple(false))
  val compType: PType = typ
  private val kcomp = kb.getCodeOrdering(typ, CodeOrdering.Compare(), ignoreMissingness = false)

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
          Region.storeAddress(storageType.fieldOffset(dest, 0), StagedRegionValueBuilder.deepCopyFromOffset(kb, region, typ, coerce[Long](v)))
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

class AppendOnlySetState(val kb: EmitClassBuilder[_], t: PType) extends PointerBasedRVAState {
  val root: Settable[Long] = kb.genFieldThisRef[Long]()
  val size: Settable[Int] = kb.genFieldThisRef[Int]()
  val key = new TypedKey(t, kb, region)
  val tree = new AppendOnlyBTree(kb, key, region, root)
  val et = EType.defaultFromPType(t)

  val typ: PStruct = PCanonicalStruct(
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

  private val _elt = kb.genFieldThisRef[Long]()
  private val _vm = kb.genFieldThisRef[Boolean]()
  private val _vv = kb.genFieldThisRef()(typeToTypeInfo(t))
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
  def foreach(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Code[Boolean], Code[_]) => Unit): Unit =
    tree.foreach(cb) { (cb, eoffCode) =>
      val eoff = cb.newLocal("casa_foreach_eoff", eoffCode)
      f(cb, key.isKeyMissing(eoff), key.loadKey(eoff))
    }

  def copyFromAddress(cb: EmitCodeBuilder, src: Code[Long]): Unit =
    cb += Code.memoize(src, "aoss_copy_from_addr_src") { src =>
      Code(
        off := region.allocate(typ.alignment, typ.byteSize),
        size := Region.loadInt(typ.loadField(src, 0)),
        tree.init,
        tree.deepCopy(Region.loadAddress(typ.loadField(src, 1))))
    }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val kEnc = et.buildEncoderMethod(t, kb)

    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      tree.bulkStore(cb, ob) { (cb, ob, srcCode) =>
        val src = cb.newLocal("aoss_ser_src", srcCode)
        cb += ob.writeBoolean(key.isKeyMissing(src))
        cb.ifx(!key.isKeyMissing(src), {
          cb += kEnc.invokeCode(key.loadKey(src), ob)
        })
      }
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val kDec = et.buildDecoderMethod(t, kb)
    val km = kb.genFieldThisRef[Boolean]("km")
    val kv = kb.genFieldThisRef("kv")(typeToTypeInfo(t))

    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb += init
      tree.bulkLoad(cb, ib) { (cb, ib, dest) =>
        cb.assign(km, ib.readBoolean())
        cb.ifx(!km, {
          cb += kv.storeAny(kDec.invokeCode(region, ib))
        })
        cb += key.store(dest, km, kv)
        cb.assign(size, size + 1)
      }
    }
  }
}

class CollectAsSetAggregator(t: PType) extends StagedAggregator {
  type State = AppendOnlySetState

  assert(t.isCanonical)
  val resultType: PSet = PCanonicalSet(t)
  val initOpTypes: Seq[PType] = Array[PType]()
  val seqOpTypes: Seq[PType] = Array[PType](t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    cb += state.init
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt) = seq
    cb += Code(elt.setup, state.insert(elt.m, elt.v))
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit =
    other.foreach(cb) { (cb, km, kv) => cb += state.insert(km, kv) }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit =
    cb += srvb.addArray(resultType.arrayFundamentalType, { sab =>
      EmitCodeBuilder.scopedVoid(cb.emb) { cb =>
        cb += sab.start(state.size)
        state.foreach(cb) { (cb, km, kv) =>
          cb.ifx(km, {
            cb += sab.setMissing()
          }, {
            cb += sab.addWithDeepCopy(t, coerce[Long](kv))
          })
          cb += sab.advance()
        }
      }
    })
}
