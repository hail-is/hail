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

  def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Code[Boolean]

  def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit

  def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit

  def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, src: Code[Long], dest: Code[Long]): Unit

  def compKeys(cb: EmitCodeBuilder, k1: EmitCode, k2: EmitCode): Code[Int]

  def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitCode

  def compSame(cb: EmitCodeBuilder, offc: Code[Long], otherc: Code[Long]): Code[Int] = {
    val off = cb.newLocal[Long]("btk_comp_same_off", offc)
    val other = cb.newLocal[Long]("btk_comp_same_other", otherc)
    compKeys(cb, loadCompKey(cb, off), loadCompKey(cb, other))
  }

  def compWithKey(cb: EmitCodeBuilder, offc: Code[Long], k: EmitCode): Code[Int] = {
    val off = cb.newLocal[Long]("btk_comp_with_key_off", offc)
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
    cb += storageType.stagedInitialize(nodeBucket, true)
    cb += elementsType.stagedInitialize(elements(nodeBucket), true)
  }

  private def isRoot(node: Code[Long]): Code[Boolean] = storageType.isFieldMissing(node, 0)

  private def isLeaf(node: Code[Long]): Code[Boolean] = storageType.isFieldMissing(node, 1)

  private def getParent(node: Code[Long]): Code[Long] = Region.loadAddress(storageType.loadField(node, 0))

  private def elements(node: Code[Long]): Code[Long] = storageType.loadField(node, 2)

  private def hasKey(node: Code[Long], i: Int): Code[Boolean] = elementsType.isFieldDefined(elements(node), i)

  private def setKeyPresent(cb: EmitCodeBuilder, node: Code[Long], i: Int): Unit = {
    cb += elementsType.setFieldPresent(elements(node), i)
  }

  private def setKeyMissing(cb: EmitCodeBuilder, node: Code[Long], i: Int): Unit = {
    cb += elementsType.setFieldMissing(elements(node), i)
  }

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

  private def setChild(cb: EmitCodeBuilder, parentC: Code[Long], i: Int, childC: Code[Long], context: String): Unit = {
    val parent = cb.newLocal[Long]("aobt_set_child_parent", parentC)
    val child = cb.newLocal[Long]("aobt_set_child_child", childC)

    if (i == -1)
      cb += storageType.setFieldPresent(parent, 1)
    cb += Region.storeAddress(childOffset(parent, i), child)
    cb += storageType.setFieldPresent(child, 0)
    cb += Region.storeAddress(storageType.fieldOffset(child, 0), parent)
  }

  private def insert(cb: EmitCodeBuilder, nodec: Code[Long], insertIdxc: Code[Int], kc: EmitCode, childC: Code[Long]): Code[Long] = {
    val kt = key.compType.sType
    val castKCode = kc.map(k => kt.coerceOrCopy(cb, region, k, false).asPCode)
    val insertAt = kb.getOrGenEmitMethod("btree_insert", (this, "insert", kt),
      FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Int], castKCode.emitParamType, typeInfo[Long]), typeInfo[Long]) { insertAt =>
      val node: Value[Long] = insertAt.getCodeParam[Long](1)
      val insertIdx: Value[Int] = insertAt.getCodeParam[Int](2)
      val k: EmitValue = insertAt.getEmitParam(3, region)
      val child: Value[Long] = insertAt.getCodeParam[Long](4)

      def parent: Code[Long] = getParent(node)

      val newNode = insertAt.newLocal[Long]()

      def makeUninitialized(cb: EmitCodeBuilder, idx: Int): Code[Long] = {
        setKeyPresent(cb, node, idx)
        key.initializeEmpty(cb, keyOffset(node, idx))
        cb.ifx(!isLeaf(node), {
          setChild(cb, node, idx, child, "makeUninitialized setChild")
        })
        loadKey(node, idx)
      }

      def copyFrom(cb: EmitCodeBuilder, destNodeC: Code[Long], destIdx: Int, srcNodeC: Code[Long], srcIdx: Int): Unit = {
        val destNode = cb.newLocal("aobt_copy_from_destnode", destNodeC)
        val srcNode = cb.newLocal("aobt_copy_from_srcnode", srcNodeC)
        setKeyPresent(cb, destNode, destIdx)
        key.copy(cb, keyOffset(srcNode, srcIdx), keyOffset(destNode, destIdx))
        cb.ifx(!isLeaf(srcNode),
          {
            setChild(cb, destNode, destIdx, loadChild(srcNode, srcIdx), "insert copyFrom")
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
          val b = cb.newLocal[Boolean]("btree_insertkey_b", !hasKey(parent, i))
          cb.ifx(!b, cb.assign(b, key.compWithKey(cb, loadKey(parent, i), ev) >= 0))
          cb.ifx(b, {
            cb.assign(upperBound, i)
            cb.goto(Lfound)
          })
        }

        cb.define(Lfound)
        cb.ifx(!isLeaf(node), {
          setChild(cb, newNode, -1, c, "insertKey !isLeaf")
        })
        cb.invokeCode(insertAt, parent, upperBound, ev, newNode)
      }


      def promote(cb: EmitCodeBuilder, idx: Int): Unit = {
        val nikey = cb.newLocal("aobt_insert_nikey", loadKey(node, idx))

        cb.ifx(!isLeaf(node), {
          setChild(cb, newNode, -1, loadChild(node, idx), "promote")
        })

        val upperBound = cb.newLocal("promote_upper_bound", maxElements)
        val Lfound = CodeLabel()

        (0 until maxElements).foreach { i =>
          val b = cb.newLocal[Boolean]("btree_insert_promote_b", !hasKey(parent, i))
          cb.ifx(!b, cb.assign(b, key.compSame(cb, loadKey(parent, i), nikey) >= 0))
          cb.ifx(b, {
            cb.assign(upperBound, i)
            cb.goto(Lfound)
          })
        }

        cb.define(Lfound)
        key.copy(cb, loadKey(node, idx), cb.invokeCode(insertAt, parent, upperBound, key.loadCompKey(cb, nikey), newNode))
        setKeyMissing(cb, node, idx)
      }

      def splitAndInsert(cb: EmitCodeBuilder): Code[Long] = {
        cb.ifx(isRoot(node), {
          createNode(cb, root)
          setChild(cb, root, -1, node, "splitAndInsert")
        })
        createNode(cb, newNode)
        val out = cb.newLocal[Long]("split_and_insert_out")
        cb.ifx(insertIdx > splitIdx, {
          copyToNew(cb, splitIdx + 1)
          promote(cb, splitIdx)
          cb.assign(out, cb.invokeCode(insertAt, newNode, insertIdx - splitIdx - 1, k, child))
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
          cb.ifx(hasKey(node, destIdx - 1), {
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
        cb.ifx(isFull(node),
          cb.assign(ret, splitAndInsert(cb)),
          cb.assign(ret, shiftAndInsert(cb)))
        ret
      }
    }

    cb.invokeCode[Long](insertAt, nodec, insertIdxc, castKCode, childC)
  }

  private def getF(cb: EmitCodeBuilder, root: Code[Long], kc: EmitCode): Code[Long] = {
    val get = kb.genEmitMethod("btree_get", FastIndexedSeq[ParamType](typeInfo[Long], kc.emitParamType), typeInfo[Long])
    get.emitWithBuilder { cb =>
      val node = get.getCodeParam[Long](1)
      val k = get.getEmitParam(2, region)

      val cmp = cb.newLocal("btree_get_cmp", -1)
      val keyV = cb.newLocal("btree_get_keyV", 0L)

      def insertOrGetAt(i: Int) = {
        cb.ifx(isLeaf(node), {
          cb.assign(keyV, insert(cb, node, const(i), k, const(0L)))
          cb.assign(cmp, 0)
        }, {
          cb.assign(node, loadChild(node, i - 1))
        })
      }

      cb.whileLoop(cmp.cne(0), { Lcont =>
        (0 until maxElements).foreach { i =>
          cb.ifx(hasKey(node, i), {
            cb.assign(keyV, loadKey(node, i))
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

  val deepCopy: (EmitCodeBuilder, Code[Long]) => Unit = {
    val f = kb.genEmitMethod("btree_deepCopy", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Long]), typeInfo[Unit])
    f.voidWithBuilder { cb =>
      val destNode = f.getCodeParam[Long](1)
      val srcNode = f.getCodeParam[Long](2)

      val er = EmitRegion(cb.emb, region)
      val newNode = cb.newLocal[Long]("new_node")

      def copyChild(i: Int): Unit = {
        createNode(cb, newNode)
        cb.invokeVoid(cb.emb, newNode, loadChild(srcNode, i))
      }

      cb.ifx(!isLeaf(srcNode), {
        copyChild(-1)
        setChild(cb, destNode, -1, newNode, "deepcopy1")
      })

      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(srcNode, i), {
          key.deepCopy(cb, er, destNode, srcNode)
          cb.ifx(!isLeaf(srcNode), {
            copyChild(i)
            setChild(cb, destNode, i, newNode, "deepcopy2")
          })
        })
      }
    }

    { (cb: EmitCodeBuilder, srcRoot: Code[Long]) => cb.invokeVoid(f, root, srcRoot) }
  }

  def bulkStore(cb: EmitCodeBuilder, obCode: Code[OutputBuffer]
  )(keyStore: (EmitCodeBuilder, Value[OutputBuffer], Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_bulkStore", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[OutputBuffer]),
      typeInfo[Unit])
    val node = f.getCodeParam[Long](1)
    val ob = f.getCodeParam[OutputBuffer](2)

    f.voidWithBuilder { cb =>
      cb += ob.writeBoolean(!isLeaf(node))
      cb.ifx(!isLeaf(node), {
        cb.invokeVoid(f, loadChild(node, -1), ob)
      })
      val Lexit = CodeLabel()
      (0 until maxElements).foreach { i =>
        cb.ifx(hasKey(node, i), {
          cb += ob.writeBoolean(true)
          keyStore(cb, ob, loadKey(node, i))
          cb.ifx(!isLeaf(node), {
            cb.invokeVoid(f, loadChild(node, i), ob)
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

  def bulkLoad(cb: EmitCodeBuilder, ibCode: Code[InputBuffer]
  )(keyLoad: (EmitCodeBuilder, Value[InputBuffer], Code[Long]) => Unit): Unit = {
    val f = kb.genEmitMethod("btree_bulkLoad", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[InputBuffer]),
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
