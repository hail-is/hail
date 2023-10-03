package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.utils._

trait BTreeKey {
  def storageType: PType

  def compType: PType

  def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean]

  def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit

  def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit

  def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, src: Code[Long], dest: Code[Long]): Unit

  def compKeys(cb: EmitCodeBuilder, k1: EmitValue, k2: EmitValue): Value[Int]

  def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitValue

  def compSame(cb: EmitCodeBuilder, offc: Code[Long], otherc: Code[Long]): Value[Int] = {
    val off = cb.newLocal[Long]("btk_comp_same_off", offc)
    val other = cb.newLocal[Long]("btk_comp_same_other", otherc)
    compKeys(cb, loadCompKey(cb, off), loadCompKey(cb, other))
  }

  def compWithKey(cb: EmitCodeBuilder, off: Value[Long], k: EmitValue): Value[Int] = {
    compKeys(cb, loadCompKey(cb, off), k)
  }
}

class AppendOnlyBTree(kb: EmitClassBuilder[_], val key: BTreeKey, region: Value[Region], root: Settable[Long], maxElements: Int = 2) {
  private val splitIdx = maxElements / 2
  private val eltType: PTuple = PCanonicalTuple(false, key.storageType, PInt64(true))
  private val elementsType: PTuple = PCanonicalTuple(required = true, Array.fill[PType](maxElements)(eltType): _*)
  private val storageType: PStruct = PCanonicalStruct(required = true,
    "parent" -> PInt64(),
    "child0" -> PInt64(),
    "elements" -> elementsType)

  private def createNode(cb: EmitCodeBuilder, nodeBucket: Settable[Long]): Unit = {
    cb.assign(nodeBucket, region.allocate(storageType.alignment, storageType.byteSize))
    storageType.stagedInitialize(cb, nodeBucket, true)
    elementsType.stagedInitialize(cb, elements(nodeBucket), true)
  }

  private def isRoot(cb: EmitCodeBuilder, node: Code[Long]): Value[Boolean] =
    storageType.isFieldMissing(cb, node, 0)

  private def isLeaf(cb: EmitCodeBuilder, node: Code[Long]): Value[Boolean] =
    storageType.isFieldMissing(cb, node, 1)

  private def getParent(cb: EmitCodeBuilder, node: Code[Long]): Value[Long] =
    cb.memoize(Region.loadAddress(storageType.loadField(node, 0)))

  private def elements(node: Code[Long]): Code[Long] = storageType.loadField(node, 2)

  private def hasKey(cb: EmitCodeBuilder, node: Code[Long], i: Int): Value[Boolean] =
    elementsType.isFieldDefined(cb, elements(node), i)

  private def setKeyPresent(cb: EmitCodeBuilder, node: Code[Long], i: Int): Unit = {
    elementsType.setFieldPresent(cb, elements(node), i)
  }

  private def setKeyMissing(cb: EmitCodeBuilder, node: Code[Long], i: Int): Unit = {
    elementsType.setFieldMissing(cb, elements(node), i)
  }

  private def isFull(cb: EmitCodeBuilder, node: Code[Long]): Value[Boolean] =
    hasKey(cb, node, maxElements - 1)

  private def keyOffset(node: Code[Long], i: Int): Code[Long] = eltType.fieldOffset(elementsType.loadField(elements(node), i), 0)

  private def loadKey(cb: EmitCodeBuilder, node: Value[Long], i: Int): Value[Long] =
    cb.memoize(eltType.loadField(elementsType.loadField(elements(node), i), 0))

  private def childOffset(node: Code[Long], i: Int): Code[Long] =
    if (i == -1)
      storageType.fieldOffset(node, 1)
    else
      eltType.fieldOffset(elementsType.loadField(elements(node), i), 1)

  private def loadChild(cb: EmitCodeBuilder, node: Code[Long], i: Int): Value[Long] =
    cb.memoize(Region.loadAddress(childOffset(node, i)))

  private def setChild(cb: EmitCodeBuilder, parentC: Code[Long], i: Int, childC: Code[Long], context: String): Unit = {
    val parent = cb.newLocal[Long]("aobt_set_child_parent", parentC)
    val child = cb.newLocal[Long]("aobt_set_child_child", childC)

    if (i == -1)
      storageType.setFieldPresent(cb, parent, 1)
    cb += Region.storeAddress(childOffset(parent, i), child)
    storageType.setFieldPresent(cb, child, 0)
    cb += Region.storeAddress(storageType.fieldOffset(child, 0), parent)
  }

  private def insert(cb: EmitCodeBuilder, nodec: Value[Long], insertIdxc: Value[Int], kc: EmitCode, childC: Value[Long]): Value[Long] = {
    val kt = key.compType.sType
    val castKCode = EmitCode.fromI(cb.emb)(cb => kc.toI(cb).map(cb)(k => kt.coerceOrCopy(cb, region, k, false)))
    val insertAt = kb.getOrGenEmitMethod("btree_insert", (this, "insert", kt),
      FastSeq[ParamType](typeInfo[Long], typeInfo[Int], castKCode.emitParamType, typeInfo[Long]), typeInfo[Long]) { insertAt =>
      val node: Value[Long] = insertAt.getCodeParam[Long](1)
      val insertIdx: Value[Int] = insertAt.getCodeParam[Int](2)
      val k: EmitValue = insertAt.getEmitParam(cb, 3)
      val child: Value[Long] = insertAt.getCodeParam[Long](4)

      def parent(cb: EmitCodeBuilder): Value[Long] = getParent(cb, node)

      val newNode = insertAt.newLocal[Long]()

      def makeUninitialized(cb: EmitCodeBuilder, idx: Int): Value[Long] = {
        setKeyPresent(cb, node, idx)
        key.initializeEmpty(cb, keyOffset(node, idx))
        cb.ifx(!isLeaf(cb, node), {
          setChild(cb, node, idx, child, "makeUninitialized setChild")
        })
        loadKey(cb, node, idx)
      }

      def copyFrom(cb: EmitCodeBuilder, destNodeC: Code[Long], destIdx: Int, srcNodeC: Code[Long], srcIdx: Int): Unit = {
        val destNode = cb.newLocal("aobt_copy_from_destnode", destNodeC)
        val srcNode = cb.newLocal("aobt_copy_from_srcnode", srcNodeC)
        setKeyPresent(cb, destNode, destIdx)
        key.copy(cb, keyOffset(srcNode, srcIdx), keyOffset(destNode, destIdx))
        cb.ifx(!isLeaf(cb, srcNode),
          {
            setChild(cb, destNode, destIdx, loadChild(cb, srcNode, srcIdx), "insert copyFrom")
          })
      }

      def copyToNew(cb: EmitCodeBuilder, startIdx: Int): Unit =
        (0 until maxElements - startIdx).foreach { newIdx =>
          copyFrom(cb, newNode, newIdx, node, newIdx + startIdx)
          setKeyMissing(cb, node, newIdx + startIdx)
        }

      def insertKey(cb: EmitCodeBuilder, ev: EmitValue, c: Code[Long]): Code[Long] = {
        val upperBound = cb.newLocal("insertKey_upper_bound", maxElements)
        val Lfound = CodeLabel()

        (0 until maxElements).foreach { i =>
          val b = cb.newLocal[Boolean]("btree_insertkey_b", !hasKey(cb, parent(cb), i))
          cb.ifx(!b, cb.assign(b, key.compWithKey(cb, loadKey(cb, parent(cb), i), ev) >= 0))
          cb.ifx(b, {
            cb.assign(upperBound, i)
            cb.goto(Lfound)
          })
        }

        cb.define(Lfound)
        cb.ifx(!isLeaf(cb, node), {
          setChild(cb, newNode, -1, c, "insertKey !isLeaf")
        })
        cb.invokeCode(insertAt, parent(cb), upperBound, ev, newNode)
      }


      def promote(cb: EmitCodeBuilder, idx: Int): Unit = {
        val nikey = cb.newLocal("aobt_insert_nikey", loadKey(cb, node, idx))

        cb.ifx(!isLeaf(cb, node), {
          setChild(cb, newNode, -1, loadChild(cb, node, idx), "promote")
        })

        val upperBound = cb.newLocal("promote_upper_bound", maxElements)
        val Lfound = CodeLabel()

        (0 until maxElements).foreach { i =>
          val b = cb.newLocal[Boolean]("btree_insert_promote_b", !hasKey(cb, parent(cb), i))
          cb.ifx(!b, cb.assign(b, key.compSame(cb, loadKey(cb, parent(cb), i), nikey) >= 0))
          cb.ifx(b, {
            cb.assign(upperBound, i)
            cb.goto(Lfound)
          })
        }

        cb.define(Lfound)
        key.copy(cb, loadKey(cb, node, idx), cb.invokeCode(insertAt, parent(cb), upperBound, key.loadCompKey(cb, nikey), newNode))
        setKeyMissing(cb, node, idx)
      }

      def splitAndInsert(cb: EmitCodeBuilder): Code[Long] = {
        cb.ifx(isRoot(cb, node), {
          createNode(cb, root)
          setChild(cb, root, -1, node, "splitAndInsert")
        })
        createNode(cb, newNode)
        val out = cb.newLocal[Long]("split_and_insert_out")
        cb.ifx(insertIdx > splitIdx, {
          copyToNew(cb, splitIdx + 1)
          promote(cb, splitIdx)
          cb.assign(out, cb.invokeCode(insertAt, newNode, cb.memoize(insertIdx - splitIdx - 1), k, child))
        }, {
          copyToNew(cb, splitIdx)
          cb.ifx(insertIdx.ceq(splitIdx), {
            cb.assign(out, insertKey(cb, k, child))
          }, {
            promote(cb, splitIdx - 1)
            cb.assign(out, cb.invokeCode(insertAt, node, insertIdx, k, child))
          })
        })
        out
      }

      def shiftAndInsert(cb: EmitCodeBuilder): Code[Long] = {
        val ret = cb.newLocal[Long]("shift_and_insert")
        val Lout = CodeLabel()
        (1 until maxElements).reverse.foreach { destIdx =>
          cb.ifx(hasKey(cb, node, destIdx - 1), {
            copyFrom(cb, node, destIdx, node, destIdx - 1)
          })
          cb.ifx(insertIdx.ceq(destIdx), {
            cb.assign(ret, makeUninitialized(cb, destIdx))
            cb.goto(Lout)
          })
        }
        cb.assign(ret, makeUninitialized(cb, 0))
        cb.define(Lout)
        ret
      }

      insertAt.emitWithBuilder { cb =>
        val ret = cb.newLocal[Long]("btree_insert_result")
        cb.ifx(isFull(cb, node),
          cb.assign(ret, splitAndInsert(cb)),
          cb.assign(ret, shiftAndInsert(cb)))
        ret
      }
    }

    cb.invokeCode[Long](insertAt, nodec, insertIdxc, castKCode, childC)
  }

  private def getF(cb: EmitCodeBuilder, root: Value[Long], kc: EmitCode): Value[Long] = {
    val get = kb.genEmitMethod("btree_get", FastSeq[ParamType](typeInfo[Long], kc.emitParamType), typeInfo[Long])
    get.emitWithBuilder { cb =>
      val node = get.getCodeParam[Long](1)
      val k = get.getEmitParam(cb, 2)

      val cmp = cb.newLocal("btree_get_cmp", -1)
      val keyV = cb.newLocal("btree_get_keyV", 0L)

      def insertOrGetAt(i: Int) = {
        cb.ifx(isLeaf(cb, node), {
          cb.assign(keyV, insert(cb, node, const(i), k, const(0L)))
          cb.assign(cmp, 0)
        }, {
          cb.assign(node, loadChild(cb, node, i - 1))
        })
      }

      cb.whileLoop(cmp.cne(0), { (Lcont: CodeLabel)  =>
        (0 until maxElements).foreach { i =>
          cb.ifx(hasKey(cb, node, i), {
            cb.assign(keyV, loadKey(cb, node, i))
            cb.assign(cmp, key.compWithKey(cb, keyV, k))
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
    cb.invokeCode(get, root, kc)
  }

  def init(cb: EmitCodeBuilder): Unit = createNode(cb, root)

  def getOrElseInitialize(cb: EmitCodeBuilder, k: EmitCode): Code[Long] = getF(cb, root, k)

  def foreach(cb: EmitCodeBuilder)(visitor: (EmitCodeBuilder, Value[Long]) => Unit): Unit = {
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
    cb.whileLoop(stackI >= 0, { (Lstart: CodeLabel) =>
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
      cb.ifx(!isLeaf(cb, node), {
        stackUpdateIdx(0)
        stackPush(loadChild(cb, node, -1))
        cb.goto(Lstart)
      })
      (0 until maxElements).foreach { i =>
        cb.define(labels(i))
        cb.ifx(hasKey(cb, node, i), {
          visitor(cb, loadKey(cb, node, i))
          cb.ifx(!isLeaf(cb, node), {
            stackUpdateIdx(i + 1)
            stackPush(loadChild(cb, node, i))
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

  val deepCopy: (EmitCodeBuilder, Value[Long]) => Unit = {
    val f = kb.genEmitMethod("btree_deepCopy", FastSeq[ParamType](typeInfo[Long], typeInfo[Long]), typeInfo[Unit])
    f.voidWithBuilder { cb =>
      val destNode = f.getCodeParam[Long](1)
      val srcNode = f.getCodeParam[Long](2)

      val er = EmitRegion(cb.emb, region)
      val newNode = cb.newLocal[Long]("new_node")

      def copyChild(i: Int): Unit = {
        createNode(cb, newNode)
        cb.invokeVoid(cb.emb, newNode, loadChild(cb, srcNode, i))
      }

      cb.ifx(!isLeaf(cb, srcNode), {
        copyChild(-1)
        setChild(cb, destNode, -1, newNode, "deepcopy1")
      })

      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(cb, srcNode, i), {
          key.deepCopy(cb, er, destNode, srcNode)
          cb.ifx(!isLeaf(cb, srcNode), {
            copyChild(i)
            setChild(cb, destNode, i, newNode, "deepcopy2")
          })
        })
      }
    }

    { (cb: EmitCodeBuilder, srcRoot: Value[Long]) => cb.invokeVoid(f, root, srcRoot) }
  }

  def bulkStore(cb: EmitCodeBuilder, obCode: Value[OutputBuffer]
  )(keyStore: (EmitCodeBuilder, Value[OutputBuffer], Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_bulkStore", FastSeq[ParamType](typeInfo[Long], typeInfo[OutputBuffer]),
      typeInfo[Unit])
    val node = f.getCodeParam[Long](1)
    val ob = f.getCodeParam[OutputBuffer](2)

    f.voidWithBuilder { cb =>
      cb += ob.writeBoolean(!isLeaf(cb, node))
      cb.ifx(!isLeaf(cb, node), {
        cb.invokeVoid(f, loadChild(cb, node, -1), ob)
      })
      val Lexit = CodeLabel()
      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(cb, node, i), {
          cb += ob.writeBoolean(true)
          keyStore(cb, ob, loadKey(cb, node, i))
          cb.ifx(!isLeaf(cb, node), {
            cb.invokeVoid(f, loadChild(cb, node, i), ob)
          })
        }, {
          cb += ob.writeBoolean(false)
          cb.goto(Lexit)
        })
      }
      cb.define(Lexit)
    }
    cb.invokeVoid(f, root, obCode)
  }

  def bulkLoad(cb: EmitCodeBuilder, ibCode: Value[InputBuffer]
  )(keyLoad: (EmitCodeBuilder, Value[InputBuffer], Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_bulkLoad", FastSeq[ParamType](typeInfo[Long], typeInfo[InputBuffer]),
      typeInfo[Unit])
    val node = f.getCodeParam[Long](1)
    val ib = f.getCodeParam[InputBuffer](2)
    val newNode = f.newLocal[Long]()
    val isInternalNode = f.newLocal[Boolean]()

    f.voidWithBuilder { cb =>
      cb.assign(isInternalNode, ib.readBoolean())
      cb.ifx(isInternalNode, {
        createNode(cb, newNode)
        setChild(cb, node, -1, newNode, "bulkLoad1")
        cb.invokeVoid(f, newNode, ib)
      })
      val Lexit = CodeLabel()
      (0 until maxElements).foreach { i =>
        cb.ifx(ib.readBoolean(), {
          setKeyPresent(cb, node, i)
          keyLoad(cb, ib, keyOffset(node, i))
          cb.ifx(isInternalNode, {
            createNode(cb, newNode)
            setChild(cb, node, i, newNode, "bulkLoad2")
            cb.invokeVoid(f, newNode, ib)
          })
        }, {
          cb.goto(Lexit)
        })
      }
      cb.define(Lexit)
    }
    cb.invokeVoid(f, root, ibCode)
  }
}
