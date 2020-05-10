package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.asm4s.coerce
import is.hail.io.{InputBuffer, OutputBuffer}

trait BTreeKey {
  def storageType: PType
  def compType: PType
  def isEmpty(off: Code[Long]): Code[Boolean]
  def initializeEmpty(off: Code[Long]): Code[Unit]

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit]
  def deepCopy(er: EmitRegion, src: Code[Long], dest: Code[Long]): Code[Unit]

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int]
  def loadCompKey(off: Value[Long]): (Code[Boolean], Code[_])

  def compSame(off: Code[Long], other: Code[Long]): Code[Int] =
    Code.memoize(off, "btk_comp_same_off", other, "btk_comp_same_other") { (off, other) =>
      compKeys(loadCompKey(off), loadCompKey(other))
    }
  def compWithKey(off: Code[Long], k: (Code[Boolean], Code[_])): Code[Int] =
    Code.memoize(off, "btk_comp_with_key_off") { off =>
      compKeys(loadCompKey(off), k)
    }
}

class AppendOnlyBTree(kb: EmitClassBuilder[_], key: BTreeKey, region: Value[Region], root: Settable[Long], maxElements: Int = 2) {
  private val splitIdx = maxElements / 2
  private val eltType: PTuple = PCanonicalTuple(false, key.storageType, PInt64(true))
  private val elementsType: PTuple = PCanonicalTuple(required = true, Array.fill[PType](maxElements)(eltType): _*)
  private val storageType: PStruct = PCanonicalStruct(required = true,
    "parent" -> PInt64(),
    "child0" -> PInt64(),
    "elements" -> elementsType)

  private def createNode(nodeBucket: Settable[Long]): Code[Unit] = Code(
    nodeBucket := region.allocate(storageType.alignment, storageType.byteSize),
    storageType.stagedInitialize(nodeBucket, true),
    elementsType.stagedInitialize(elements(nodeBucket), true))

  private def isRoot(node: Code[Long]): Code[Boolean] = storageType.isFieldMissing(node, 0)
  private def isLeaf(node: Code[Long]): Code[Boolean] = storageType.isFieldMissing(node, 1)
  private def getParent(node: Code[Long]): Code[Long] = Region.loadAddress(storageType.loadField(node, 0))

  private def elements(node: Code[Long]): Code[Long] = storageType.loadField(node, 2)
  private def hasKey(node: Code[Long], i: Int): Code[Boolean] = elementsType.isFieldDefined(elements(node), i)
  private def setKeyPresent(node: Code[Long], i: Int): Code[Unit] = elementsType.setFieldPresent(elements(node), i)
  private def setKeyMissing(node: Code[Long], i: Int): Code[Unit] = elementsType.setFieldMissing(elements(node), i)
  private def isFull(node: Code[Long]): Code[Boolean] = hasKey(node, maxElements - 1)
  private def keyOffset(node: Code[Long], i: Int): Code[Long] = eltType.fieldOffset(elementsType.loadField(elements(node), i), 0)
  private def loadKey(node: Code[Long], i: Int): Code[Long] = eltType.loadField(elementsType.loadField(elements(node), i), 0)

  private def childOffset(node: Code[Long], i: Int): Code[Long] =
    if (i == -1)
      storageType.fieldOffset(node, 1)
    else
      eltType.fieldOffset(elementsType.loadField(elements(node), i), 1)
  private def loadChild(node: Code[Long], i: Int): Code[Long] =
    Region.loadAddress(childOffset(node, i))
  private def setChild(parent: Code[Long], i: Int, child: Code[Long]): Code[Unit] =
    Code.memoize(parent, "aobt_set_child_parent",
      child, "aobt_set_child_child") { (parent, child) =>
      Code(
        if (i == -1) storageType.setFieldPresent(parent, 1) else Code._empty,
        Region.storeAddress(childOffset(parent, i), child),
        storageType.setFieldPresent(child, 0),
        Region.storeAddress(storageType.fieldOffset(child, 0), parent))
    }

  private val insert: EmitMethodBuilder[_] = {
    val insertAt = kb.genEmitMethod("btree_insert", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Int], typeInfo[Boolean], key.compType.ti, typeInfo[Long]), typeInfo[Long])
    val node: Value[Long] = insertAt.getCodeParam[Long](1)
    val insertIdx: Value[Int] = insertAt.getCodeParam[Int](2)
    val km: Value[Boolean] = insertAt.getCodeParam[Boolean](3)
    val kv: Value[_] = insertAt.getCodeParam(4)(key.compType.ti)
    val child: Value[Long] = insertAt.getCodeParam[Long](5)

    def parent: Code[Long] = getParent(node)

    val newNode = insertAt.newLocal[Long]()

    def makeUninitialized(idx: Int): Code[Long] =
      Code(
        setKeyPresent(node, idx),
        key.initializeEmpty(keyOffset(node, idx)),
        (!isLeaf(node)).orEmpty(setChild(node, idx, child)),
        loadKey(node, idx))

    def copyFrom(destNode: Code[Long], destIdx: Int, srcNode: Code[Long], srcIdx: Int): Code[Unit] =
      Code.memoize(destNode, "aobt_copy_from_destnode",
        srcNode, "aobt_copy_from_srcnode") { (destNode, srcNode) =>
        Code(
          setKeyPresent(destNode, destIdx),
          key.copy(keyOffset(srcNode, srcIdx), keyOffset(destNode, destIdx)),
          (!isLeaf(srcNode)).orEmpty(setChild(destNode, destIdx, loadChild(srcNode, srcIdx))))
      }

    val shiftAndInsert = Array.range(1, maxElements)
      .foldLeft(makeUninitialized(0)) { (cont, destIdx) =>
        Code(
          hasKey(node, destIdx - 1).orEmpty(
            copyFrom(node, destIdx, node, destIdx - 1)),
          insertIdx.ceq(destIdx).mux(
            makeUninitialized(destIdx),
            cont))
      }

    def copyToNew(startIdx: Int) = Code(
      Array.tabulate(maxElements - startIdx) { newIdx =>
        Code(copyFrom(newNode, newIdx, node, newIdx + startIdx),
          setKeyMissing(node, newIdx + startIdx))
      })

    def insertKey(m: Value[Boolean], v: Value[_], c: Code[Long]): Code[Long] = {
      val upperBound = Array.range(0, maxElements)
        .foldRight(maxElements: Code[Int]) { (i, cont) =>
          (!hasKey(parent, i) ||
            key.compWithKey(loadKey(parent, i), m.get -> v.get) >= 0)
            .mux(i, cont)
        }
      Code((!isLeaf(node)).orEmpty(
        setChild(newNode, -1, c)),
        insertAt.invokeCode[Long](parent, upperBound, m, v, newNode))
    }

    def promote(idx: Int): Code[Unit] = {
      val upperBound = Array.range(0, maxElements)
        .foldRight(maxElements: Code[Int]) { (i, cont) =>
          (!hasKey(parent, i) ||
            key.compSame(loadKey(parent, i), loadKey(node, idx)) >= 0)
            .mux(i, cont)
        }
      Code.memoize(loadKey(node, idx), "aobt_insert_nikey") { nikey =>
        val (compKeyM, compKeyV) = key.loadCompKey(nikey)
        Code((!isLeaf(node)).orEmpty(
          setChild(newNode, -1, loadChild(node, idx))),
          key.copy(loadKey(node, idx),
            insertAt.invokeCode[Long](parent, upperBound, compKeyM, compKeyV, newNode)),
          setKeyMissing(node, idx))
      }
    }

    val splitAndInsert = Code(
      isRoot(node).orEmpty(Code(
        createNode(root),
        setChild(root, -1, node))),
      createNode(newNode),
      (insertIdx > splitIdx).mux(
        Code(copyToNew(splitIdx + 1),
          promote(splitIdx),
          insertAt.invokeCode(newNode, insertIdx - splitIdx - 1, km, kv, child)),
        Code(
          copyToNew(splitIdx),
          insertIdx.ceq(splitIdx).mux(
            insertKey(km, kv, child),
            Code(promote(splitIdx - 1),
              insertAt.invokeCode(node, insertIdx, km, kv, child))))))

    insertAt.emit(isFull(node).mux(splitAndInsert, shiftAndInsert))
    insertAt
  }

  private val getF: EmitMethodBuilder[_] = {
    val get = kb.genEmitMethod("btree_get", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Boolean], key.compType.ti), typeInfo[Long])
    val node = get.getCodeParam[Long](1)
    val km = get.getCodeParam[Boolean](2)
    val kv = get.getCodeParam(3)(key.compType.ti)

    val cmp = get.newLocal[Int]()
    val keyV = get.newLocal[Long]()

    def insertOrGetAt(i: Int) = isLeaf(node).mux(
      Code(keyV := insert.invokeCode[Long](node, const(i), km, kv, const(0L)), cmp := 0),
      node := loadChild(node, i - 1))

    get.emit(Code(
      cmp := -1,
      keyV := 0L,
      Code.whileLoop(cmp.cne(0),
        Array.range(0, maxElements).foldRight(insertOrGetAt(maxElements)) { (i, cont) =>
          val storeKeyAndComp = Code(
            keyV := loadKey(node, i),
            cmp := key.compWithKey(keyV, km.get -> kv.get))
          (hasKey(node, i) && Code(storeKeyAndComp, cmp <= 0)).mux(
            (cmp < 0).orEmpty(cont),
            insertOrGetAt(i))
        }),
      keyV))
    get
  }

  def init: Code[Unit] = createNode(root)

  def getOrElseInitialize(km: Code[Boolean], kv: Code[_]): Code[Long] =
    getF.invokeCode(root, km, kv)

  def foreach(cb: EmitCodeBuilder)(visitor: (EmitCodeBuilder, Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_foreach", FastIndexedSeq[ParamType](typeInfo[Long]), typeInfo[Unit])
    val node = f.getCodeParam[Long](1)
    val i = f.newLocal[Int]("aobt_foreach_i")

    f.emitWithBuilder { cb =>
      cb.ifx(!isLeaf(node), {
        cb += f.invokeCode(loadChild(node, -1))
      })
      cb.assign(i, 0)
      val Lexit = CodeLabel()
      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(node, i), {
          visitor(cb, loadKey(node, i))
          cb.ifx(!isLeaf(node), {
            cb += f.invokeCode(loadChild(node, i))
          })
        }, {
          cb.goto(Lexit)
        })
      }
      cb.define(Lexit)
      Code._empty
    }
    cb += f.invokeCode(root)
  }

  val deepCopy: Code[Long] => Code[Unit] = {
    val f = kb.genEmitMethod("btree_deepCopy", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Long]), typeInfo[Unit])
    val destNode = f.getCodeParam[Long](1)
    val srcNode = f.getCodeParam[Long](2)

    val er = EmitRegion(f, region)
    val newNode = f.newLocal[Long]()

    def copyChild(i: Int) =
      Code(createNode(newNode),
        f.invokeCode[Unit](newNode, loadChild(srcNode, i)))

    val copyNodes = Array.range(0, maxElements).foldRight(Code._empty) { (i, cont) =>
      hasKey(srcNode, i).orEmpty(
        Code(
          key.deepCopy(er, destNode, srcNode),
          (!isLeaf(srcNode)).orEmpty(Code(
            copyChild(i),
            setChild(destNode, i, newNode))),
          cont))
    }

    f.emit(Code(
      (!isLeaf(srcNode)).orEmpty(
        Code(copyChild(-1),
          setChild(destNode, -1, newNode))),
      copyNodes))

    { srcRoot: Code[Long] => f.invokeCode(root, srcRoot) }
  }

  def bulkStore(cb: EmitCodeBuilder, obCode: Code[OutputBuffer]
  )(keyStore: (EmitCodeBuilder, Value[OutputBuffer], Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_bulkStore", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[OutputBuffer]),
      typeInfo[Unit])
    val node = f.getCodeParam[Long](1)
    val ob = f.getCodeParam[OutputBuffer](2)

    f.emitWithBuilder { cb =>
      cb += ob.writeBoolean(!isLeaf(node))
      cb.ifx(!isLeaf(node), {
        cb += f.invokeCode(loadChild(node, -1), ob)
      })
      val Lexit = CodeLabel()
      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(node, i), {
          cb += ob.writeBoolean(true)
          keyStore(cb, ob, loadKey(node, i))
          cb.ifx(!isLeaf(node), {
            cb += f.invokeCode(loadChild(node, i), ob)
          })
        }, {
          cb += ob.writeBoolean(false)
          cb.goto(Lexit)
        })
      }
      cb.define(Lexit)
      Code._empty
    }
    cb += f.invokeCode(root, obCode)
  }

  def bulkLoad(cb: EmitCodeBuilder, ibCode: Code[InputBuffer]
  )(keyLoad: (EmitCodeBuilder, Value[InputBuffer], Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_bulkLoad", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[InputBuffer]),
      typeInfo[Unit])
    val node = f.getCodeParam[Long](1)
    val ib = f.getCodeParam[InputBuffer](2)
    val newNode = f.newLocal[Long]()
    val isInternalNode = f.newLocal[Boolean]()

    f.emitWithBuilder { cb =>
      cb.assign(isInternalNode, ib.readBoolean())
      cb.ifx(isInternalNode, {
        cb += createNode(newNode)
        cb += setChild(node, -1, newNode)
        cb += f.invokeCode(newNode, ib)
      })
      val Lexit = CodeLabel()
      (0 until maxElements).foreach { i =>
        cb.ifx(ib.readBoolean(), {
          cb += setKeyPresent(node, i)
          keyLoad(cb, ib, keyOffset(node, i))
          cb.ifx(isInternalNode, {
            cb += createNode(newNode)
            cb += setChild(node, i, newNode)
            cb += f.invokeCode(newNode, ib)
          })
        }, {
          cb.goto(Lexit)
        })
      }
      cb.define(Lexit)
      Code._empty
    }
    cb += f.invokeCode(root, ibCode)
  }
}
