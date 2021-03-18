package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitRegion, IEmitCode}
import is.hail.io._
import is.hail.types.VirtualTypeWithReq
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.virtual.Type
import is.hail.utils._

class TypedKey(typ: PType, kb: EmitClassBuilder[_], region: Value[Region]) extends BTreeKey {
  val storageType: PTuple = PCanonicalTuple(false, typ, PCanonicalTuple(false))
  val compType: PType = typ

  def isKeyMissing(src: Code[Long]): Code[Boolean] = storageType.isFieldMissing(src, 0)

  def loadKey(cb: EmitCodeBuilder, src: Code[Long]): PCode = {
    typ.loadCheapPCode(cb, storageType.loadField(src, 0))
  }

  def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Code[Boolean] = storageType.isFieldMissing(off, 1)

  def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit =
    cb += storageType.setFieldMissing(off, 1)

  def store(cb: EmitCodeBuilder, destc: Code[Long], k: EmitCode): Unit = {
    val dest = cb.newLocal("casa_store_dest", destc)

    cb += storageType.setFieldPresent(dest, 1)
    k.toI(cb)
      .consume(cb,
        {
          cb += storageType.setFieldMissing(dest, 0)
        },
        { sc =>
          cb += storageType.setFieldPresent(dest, 0)
          typ.storeAtAddress(cb, storageType.fieldOffset(dest, 0), region, sc, deepCopy = true)
        })
  }

  def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit =
    cb += Region.copyFrom(src, dest, storageType.byteSize)

  def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, dest: Code[Long], src: Code[Long]): Unit = {
    storageType.storeAtAddress(cb, dest, region, storageType.loadCheapPCode(cb, src), deepCopy = true)
  }

  def compKeys(cb: EmitCodeBuilder, k1: EmitCode, k2: EmitCode): Code[Int] = {
    kb.getOrderingFunction(k1.st, k2.st, CodeOrdering.Compare())(cb, k1, k2)
  }

  def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitCode =
    EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, isKeyMissing(off), loadKey(cb, off)))
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

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, srcc: Code[Long]): Unit = {
    super.load(cb, regionLoader, srcc)
    cb.ifx(off.cne(0L),
      {
        cb.assign(size, Region.loadInt(typ.loadField(off, 0)))
        cb.assign(root, Region.loadAddress(typ.loadField(off, 1)))
      })
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    cb += Region.storeInt(typ.fieldOffset(off, 0), size)
    cb += Region.storeAddress(typ.fieldOffset(off, 1), root)
    super.store(cb, regionStorer, destc)
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
    cb.ifx(key.isEmpty(cb, _elt), {
      cb.assign(size, size + 1)
      key.store(cb, _elt, _v)
    })
  }

  // loads container; does not update.
  def foreach(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, EmitCode) => Unit): Unit =
    tree.foreach(cb) { (cb, eoffCode) =>
      val eoff = cb.newLocal("casa_foreach_eoff", eoffCode)
      f(cb, EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, key.isKeyMissing(eoff), key.loadKey(cb, eoff))))
    }

  def copyFromAddress(cb: EmitCodeBuilder, srcc: Code[Long]): Unit = {
    val src = cb.newLocal[Long]("aoss_copy_from_addr_src", srcc)
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    cb.assign(size, Region.loadInt(typ.loadField(src, 0)))
    tree.init(cb)
    tree.deepCopy(cb, Region.loadAddress(typ.loadField(src, 1)))
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      tree.bulkStore(cb, ob) { (cb, ob, srcCode) =>
        val src = cb.newLocal("aoss_ser_src", srcCode)
        cb += ob.writeBoolean(key.isKeyMissing(src))
        cb.ifx(!key.isKeyMissing(src), {
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
  val resultType: PCanonicalSet = PCanonicalSet(elemPType)
  private[this] val arrayRep = resultType.arrayRep
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

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    other.foreach(cb) { (cb, k) => state.insert(cb, k) }
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    assert(pt == resultType)
    val (pushElement, finish) = arrayRep.constructFromFunctions(cb, region, state.size, deepCopy = true)
    state.foreach(cb) { (cb, elt) =>
      pushElement(cb, elt.toI(cb))
    }
    // deepCopy is handled by `storeElement` above
    resultType.storeAtAddress(cb, addr, region, finish(cb), deepCopy = false)
  }
}
