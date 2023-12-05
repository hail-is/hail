package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitRegion, EmitValue, IEmitCode}
import is.hail.io._
import is.hail.types.VirtualTypeWithReq
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.virtual.Type
import is.hail.utils._

class TypedKey(typ: PType, kb: EmitClassBuilder[_], region: Value[Region]) extends BTreeKey {
  override val storageType: PTuple = PCanonicalTuple(false, typ, PCanonicalTuple(false))
  override val compType: PType = typ

  def isKeyMissing(cb: EmitCodeBuilder, src: Code[Long]): Value[Boolean] =
    storageType.isFieldMissing(cb, src, 0)

  def loadKey(cb: EmitCodeBuilder, src: Code[Long]): SValue = {
    typ.loadCheapSCode(cb, storageType.loadField(src, 0))
  }

  override def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    storageType.isFieldMissing(cb, off, 1)

  override def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit =
    storageType.setFieldMissing(cb, off, 1)

  def store(cb: EmitCodeBuilder, destc: Code[Long], k: EmitCode): Unit = {
    val dest = cb.newLocal("casa_store_dest", destc)

    storageType.setFieldPresent(cb, dest, 1)
    k.toI(cb)
      .consume(cb,
        {
          storageType.setFieldMissing(cb, dest, 0)
        },
        { sc =>
          storageType.setFieldPresent(cb, dest, 0)
          typ.storeAtAddress(cb, storageType.fieldOffset(dest, 0), region, sc, deepCopy = true)
        })
  }

  override def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit =
    cb += Region.copyFrom(src, dest, storageType.byteSize)

  override def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, dest: Code[Long], src: Code[Long]): Unit = {
    storageType.storeAtAddress(cb, dest, region, storageType.loadCheapSCode(cb, src), deepCopy = true)
  }

  override def compKeys(cb: EmitCodeBuilder, k1: EmitValue, k2: EmitValue): Value[Int] = {
    kb.getOrderingFunction(k1.st, k2.st, CodeOrdering.Compare())(cb, k1, k2)
  }

  override def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitValue =
    EmitValue(Some(isKeyMissing(cb, off)), loadKey(cb, off))
}

class AppendOnlySetState(val kb: EmitClassBuilder[_], vt: VirtualTypeWithReq) extends PointerBasedRVAState {
  private val t = vt.canonicalPType
  val root: Settable[Long] = kb.genFieldThisRef[Long]()
  val size: Settable[Int] = kb.genFieldThisRef[Int]()
  val key = new TypedKey(t, kb, region)
  val tree = new AppendOnlyBTree(kb, key, region, root)
  val et = EType.defaultFromPType(t)

  val typ: PStruct = PCanonicalStruct(
    required = true,
    "size" -> PInt32(true),
    "tree" -> PInt64(true))

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    super.load(cb, regionLoader, src)
    cb.if_(off.cne(0L),
      {
        cb.assign(size, Region.loadInt(typ.loadField(off, 0)))
        cb.assign(root, Region.loadAddress(typ.loadField(off, 1)))
      })
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, dest: Value[Long]): Unit = {
    cb += Region.storeInt(typ.fieldOffset(off, 0), size)
    cb += Region.storeAddress(typ.fieldOffset(off, 1), root)
    super.store(cb, regionStorer, dest)
  }

  def init(cb: EmitCodeBuilder): Unit = {
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    cb.assign(size, 0)
    tree.init(cb)
  }

  private val _elt = kb.genFieldThisRef[Long]()

  def insert(cb: EmitCodeBuilder, v: EmitCode): Unit = {
    val _v = cb.memoize(v, "collect_as_set_insert_value")
    cb.assign(_elt, tree.getOrElseInitialize(cb, _v))
    cb.if_(key.isEmpty(cb, _elt), {
      cb.assign(size, size + 1)
      key.store(cb, _elt, _v)
    })
  }

  // loads container; does not update.
  def foreach(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, EmitCode) => Unit): Unit =
    tree.foreach(cb) { (cb, eoffCode) =>
      val eoff = cb.newLocal("casa_foreach_eoff", eoffCode)
      f(cb, EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, key.isKeyMissing(cb, eoff), key.loadKey(cb, eoff))))
    }

  def copyFromAddress(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    cb.assign(size, Region.loadInt(typ.loadField(src, 0)))
    tree.init(cb)
    tree.deepCopy(cb, cb.memoize(Region.loadAddress(typ.loadField(src, 1))))
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      tree.bulkStore(cb, ob) { (cb, ob, srcCode) =>
        val src = cb.newLocal("aoss_ser_src", srcCode)
        cb += ob.writeBoolean(key.isKeyMissing(cb, src))
        cb.if_(!key.isKeyMissing(cb, src), {
          val k = key.loadKey(cb, src)
          et.buildEncoder(k.st, kb)
              .apply(cb, k, ob)
        })
      }
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val kDec = et.buildDecoder(t.virtualType, kb)
    val km = kb.genFieldThisRef[Boolean]("km")
    val kv = kb.genFieldThisRef("kv")(typeToTypeInfo(t))

    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      init(cb)
      tree.bulkLoad(cb, ib) { (cb, ib, dest) =>
        val km = cb.newLocal[Boolean]("collect_as_set_deser_km", ib.readBoolean())
        key.store(cb, dest, EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, km, kDec(cb, region, ib))))
        cb.assign(size, size + 1)
      }
    }
  }
}

class CollectAsSetAggregator(elem: VirtualTypeWithReq) extends StagedAggregator {
  type State = AppendOnlySetState

  private val elemPType = elem.canonicalPType
  val setPType = PCanonicalSet(elemPType)
  val setSType = SIndexablePointer(setPType)
  val resultEmitType: EmitType = EmitType(setSType, true)
  private[this] val arrayRep = resultEmitType.storageType.asInstanceOf[PCanonicalSet].arrayRep
  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](elem.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    state.init(cb)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt) = seq
    state.insert(cb, elt)
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, region: Value[Region], state: AppendOnlySetState, other: AppendOnlySetState): Unit = {
    other.foreach(cb) { (cb, k) => state.insert(cb, k) }
  }

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    val (pushElement, finish) = arrayRep.constructFromFunctions(cb, region, state.size, deepCopy = true)
    state.foreach(cb) { (cb, elt) =>
      pushElement(cb, elt.toI(cb))
    }
    assert(arrayRep.required)
    // deepCopy is handled by `storeElement` above
    IEmitCode.present(cb, setPType.construct(finish(cb)))
  }
}
