package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical._
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

  def compKeys(k1: EmitCode, k2: EmitCode): Code[Int]
  def loadCompKey(off: Value[Long]): EmitCode

  def compSame(off: Code[Long], other: Code[Long]): Code[Int] =
    Code.memoize(off, "btk_comp_same_off", other, "btk_comp_same_other") { (off, other) =>
      compKeys(loadCompKey(off), loadCompKey(other))
    }
  def compWithKey(off: Code[Long], k: EmitCode): Code[Int] =
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
    val insertAt = kb.genEmitMethod("btree_insert", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Int], key.compType.asEmitParam, typeInfo[Long]), typeInfo[Long])
    val node: Value[Long] = insertAt.getCodeParam[Long](1)
    val insertIdx: Value[Int] = insertAt.getCodeParam[Int](2)
    val k: EmitValue = insertAt.getEmitParam(3)
    val child: Value[Long] = insertAt.getCodeParam[Long](4)

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

    def insertKey(ev: EmitValue, c: Code[Long]): Code[Long] = EmitCodeBuilder.scopedCode(insertAt) { cb =>
      val upperBound = Array.range(0, maxElements)
        .foldRight(maxElements: Code[Int]) { (i, cont) =>
          (!hasKey(parent, i) ||
            key.compWithKey(loadKey(parent, i), ev) >= 0)
            .mux(i, cont)
        }

      cb.ifx(!isLeaf(node), {
        cb += setChild(newNode, -1, c)
      })
      cb.invokeCode(insertAt, parent, upperBound, ev, newNode)
    }

    def promote(idx: Int): Code[Unit] = EmitCodeBuilder.scopedVoid(insertAt) { cb =>
      cb.ifx(!isLeaf(node), {
        cb += setChild(newNode, -1, loadChild(node, idx))
      })

      val upperBound = cb.newLocal("promote_upper_bound", maxElements)
      val Lfound = CodeLabel()

      (0 until maxElements).foreach { i =>
        cb.ifx(!hasKey(parent, i) || key.compSame(loadKey(parent, i), loadKey(node, idx)) >= 0, {
          cb.assign(upperBound, i)
          cb.goto(Lfound)
        })
      }

      cb.define(Lfound)
      val nikey = cb.newLocal("aobt_insert_nikey", loadKey(node, idx))
      val compKey = key.loadCompKey(nikey)
      cb += key.copy(loadKey(node, idx), cb.invokeCode(insertAt, parent, upperBound, compKey, newNode))
      cb += setKeyMissing(node, idx)
    }

    val splitAndInsert = EmitCodeBuilder.scopedCode(insertAt) { cb =>
      cb.ifx(isRoot(node), {
        cb += createNode(root)
        cb += setChild(root, -1, node)
      })
      cb += createNode(newNode)
      val out = cb.newLocal[Long]("split_and_insert_out")
      cb.ifx(insertIdx > splitIdx, {
        cb += copyToNew(splitIdx + 1)
        cb += promote(splitIdx)
        cb.assign(out, cb.invokeCode(insertAt, newNode, insertIdx - splitIdx - 1, k, child))
      }, {
        cb += copyToNew(splitIdx)
        cb.ifx(insertIdx.ceq(splitIdx), {
          cb.assign(out, insertKey(k, child))
        }, {
          cb += promote(splitIdx - 1)
          cb.assign(out, cb.invokeCode(insertAt, node, insertIdx, k, child))
        })
      })
      out
    }

    insertAt.emit(isFull(node).mux(splitAndInsert, shiftAndInsert))
    insertAt
  }

  private val getF: EmitMethodBuilder[_] = {
    val get = kb.genEmitMethod("btree_get", FastIndexedSeq[ParamType](typeInfo[Long], key.compType.asEmitParam), typeInfo[Long])
    val node = get.getCodeParam[Long](1)
    val k = get.getEmitParam(2)

    get.emitWithBuilder { cb =>
      val cmp = cb.newLocal("cmp", -1)
      val keyV = cb.newLocal("keyV", 0L)

      def insertOrGetAt(i: Int) = {
        cb.ifx(isLeaf(node), {
          cb.assign(keyV, cb.invokeCode(insert, node, const(i), k, const(0L)))
          cb.assign(cmp, 0)
        }, {
          cb.assign(node, loadChild(node, i - 1))
        })
      }

      cb.whileLoop(cmp.cne(0), { Lcont =>
        (0 until maxElements).foreach { i =>
          cb.ifx(hasKey(node, i), {
            cb.assign(keyV, loadKey(node, i))
            cb.assign(cmp, key.compWithKey(keyV, k))
            cb.ifx(cmp.ceq(0), cb.goto(Lcont))
            cb.ifx(cmp > 0, {
              insertOrGetAt(i)
              cb.goto(Lcont)
            })
          }, {
            insertOrGetAt(i)
            cb.goto(Lcont)
          })
        }
        insertOrGetAt(maxElements)
      })
      keyV.get
    }
    get
  }

  def init: Code[Unit] = createNode(root)

  def getOrElseInitialize(cb: EmitCodeBuilder, k: EmitCode): Code[Long] =
    cb.invokeCode(getF, root, k)

  def foreach(cb: EmitCodeBuilder)(visitor: (EmitCodeBuilder, Code[Long]) => Unit): Unit = {
    val stackI = cb.newLocal[Int]("btree_foreach_stack_i", -1)
    val nodeStack = cb.newLocal("btree_foreach_node_stack", Code.newArray[Long](const(128)))
    val idxStack = cb.newLocal("btree_foreach_index_stack", Code.newArray[Int](const(128)))

    def stackPush(node: Code[Long]) = {
      cb.assign(stackI, stackI + 1)
      cb += nodeStack.update(stackI, node)
      cb += idxStack.update(stackI, -1)
    }
    def stackUpdateIdx(newIdx: Code[Int]) = {
      cb += idxStack.update(stackI, newIdx)
    }
    def stackPop() = {
      cb.assign(stackI, stackI - 1)
    }

    stackPush(root)
    cb.whileLoop(stackI >= 0, { (Lstart) =>
      val node = cb.newLocal("btree_foreach_node", nodeStack(stackI))
      val idx = cb.newLocal("btree_foreach_idx", idxStack(stackI))
      val Lend = CodeLabel()
      val Lminus1 = CodeLabel()
      val labels = Array.fill[CodeLabel](maxElements)(CodeLabel())
      // this should probably be a switch, don't know how to make it one though
      // furthermore, we should be able to do the lookups at runtime
      // FIXME, clean this up once we have fixed arrays
      cb.ifx(idx.ceq(-1), cb.goto(Lminus1))
      (0 until maxElements).zip(labels).foreach { case (i, l) =>
        cb.ifx(idx.ceq(i), cb.goto(l))
      }
      cb.goto(Lend)

      cb.define(Lminus1)
      cb.ifx(!isLeaf(node), {
        stackUpdateIdx(0)
        stackPush(loadChild(node, -1))
        cb.goto(Lstart)
      })
      (0 until maxElements).foreach { i =>
        cb.define(labels(i))
        cb.ifx(hasKey(node, i), {
          visitor(cb, loadKey(node, i))
          cb.ifx(!isLeaf(node), {
            stackUpdateIdx(i + 1)
            stackPush(loadChild(node, i))
            cb.goto(Lstart)
          })
        }, {
          cb.goto(Lend)
        })
      }

      cb.define(Lend)
      stackPop()
    })
  }

  val deepCopy: Code[Long] => Code[Unit] = {
    val f = kb.genEmitMethod("btree_deepCopy", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Long]), typeInfo[Unit])
    f.voidWithBuilder { cb =>
      val destNode = f.getCodeParam[Long](1)
      val srcNode = f.getCodeParam[Long](2)

      val er = EmitRegion(cb.emb, region)
      val newNode = cb.newLocal[Long]("new_node")

      def copyChild(i: Int): Unit = {
        cb += createNode(newNode)
        cb.invokeVoid(cb.emb, newNode, loadChild(srcNode, i))
      }

      cb.ifx(!isLeaf(srcNode), {
        copyChild(-1)
        cb += setChild(destNode, -1, newNode)
      })

      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(srcNode, i), {
          cb += key.deepCopy(er, destNode, srcNode)
          cb.ifx(!isLeaf(srcNode), {
            copyChild(i)
            cb += setChild(destNode, i, newNode)
          })
        })
      }
    }

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
