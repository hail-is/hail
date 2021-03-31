package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SIndexablePointerCode, SIndexablePointerSettable}
import is.hail.utils._

object StagedBlockLinkedList {
  val defaultBlockCap: Int = 64

  private val nil: Long = 0L
}

class StagedBlockLinkedList(val elemType: PType, val kb: EmitClassBuilder[_]) {

  import StagedBlockLinkedList._

  private val firstNode = kb.genFieldThisRef[Long]()
  private val lastNode = kb.genFieldThisRef[Long]()
  private val totalCount = kb.genFieldThisRef[Int]()

  val storageType = PCanonicalStruct(
    "firstNode" -> PInt64Required,
    "lastNode" -> PInt64Required,
    "totalCount" -> PInt32Required)

  def load(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    cb += Code.memoize(src, "sbll_load_src") { src =>
      Code(
        firstNode := Region.loadAddress(storageType.fieldOffset(src, "firstNode")),
        lastNode := Region.loadAddress(storageType.fieldOffset(src, "lastNode")),
        totalCount := Region.loadInt(storageType.fieldOffset(src, "totalCount")))
    }
  }

  def store(cb: EmitCodeBuilder, dst: Code[Long]): Unit =
    cb += Code.memoize(dst, "sbll_store_dst") { dst =>
      Code(
        Region.storeAddress(storageType.fieldOffset(dst, "firstNode"), firstNode),
        Region.storeAddress(storageType.fieldOffset(dst, "lastNode"), lastNode),
        Region.storeInt(storageType.fieldOffset(dst, "totalCount"), totalCount))
    }

  type Node = Value[Long]

  val bufferType = PCanonicalArray(elemType, required = true)
  val bufferEType = EArray(EType.defaultFromPType(elemType), required = true)

  val nodeType = PCanonicalStruct(
    "buf" -> bufferType,
    "count" -> PInt32Required,
    "next" -> PInt64Required)

  private def buffer(n: Node): Code[Long] =
    Region.loadAddress(nodeType.fieldOffset(n, "buf"))

  private[agg] def capacity(n: Node): Code[Int] =
    bufferType.loadLength(buffer(n))

  private def count(n: Node): Code[Int] =
    Region.loadInt(nodeType.fieldOffset(n, "count"))

  private def incrCount(n: Node): Code[Unit] =
    Region.storeInt(nodeType.fieldOffset(n, "count"), count(n) + 1)

  private def next(n: Node): Code[Long] =
    Region.loadAddress(nodeType.fieldOffset(n, "next"))

  private def hasNext(n: Node): Code[Boolean] =
    next(n) cne nil

  private def setNext(n: Node, nNext: Node): Code[Unit] =
    Region.storeAddress(nodeType.fieldOffset(n, "next"), nNext)

  private def initNode(n: Node, buf: Code[Long], count: Code[Int]): Code[Unit] =
    Code(
      Region.storeAddress(nodeType.fieldOffset(n, "buf"), buf),
      Region.storeInt(nodeType.fieldOffset(n, "count"), count),
      Region.storeAddress(nodeType.fieldOffset(n, "next"), nil))

  private def pushPresent(cb: EmitCodeBuilder, n: Node)(store: (EmitCodeBuilder, Code[Long]) => Unit): Unit = {
    cb += bufferType.setElementPresent(buffer(n), count(n))
    store(cb, bufferType.elementOffset(buffer(n), capacity(n), count(n)))
    cb += incrCount(n)
  }

  private def pushMissing(n: Node): Code[Unit] =
    Code(
      bufferType.setElementMissing(buffer(n), count(n)),
      incrCount(n))

  private def allocateNode(dstNode: Settable[Long])(r: Value[Region], cap: Code[Int]): Code[Unit] =
    Code.memoize(cap, "sbll_alloc_node_cap") { cap =>
      Code(
        dstNode := r.allocate(nodeType.alignment, nodeType.byteSize),
        initNode(dstNode,
          buf = bufferType.allocate(r, cap),
          count = 0),
        bufferType.stagedInitialize(buffer(dstNode), cap))
    }

  private def initWithCapacity(r: Value[Region], initialCap: Code[Int]): Code[Unit] = {
    Code(
      allocateNode(firstNode)(r, initialCap),
      lastNode := firstNode,
      totalCount := 0)
  }

  def init(cb: EmitCodeBuilder, r: Value[Region]): Unit =
    cb += initWithCapacity(r, defaultBlockCap)

  private def pushNewBlockNode(mb: EmitMethodBuilder[_], r: Value[Region], cap: Code[Int]): Code[Unit] = {
    val newNode = mb.newLocal[Long]()
    Code(
      allocateNode(newNode)(r, cap),
      setNext(lastNode, newNode),
      lastNode := newNode)
  }

  private def foreachNode(cb: EmitCodeBuilder, tmpNode: Settable[Long])(body: EmitCodeBuilder => Unit): Unit = {
    val present = cb.newLocal[Boolean]("bll_foreachnode_present")
    cb.assign(tmpNode, firstNode)
    cb.assign(present, true)
    cb.whileLoop(present,
      {
        body(cb)
        cb.assign(present, hasNext(tmpNode))
        cb.assign(tmpNode, next(tmpNode))
      })
  }

  private def foreach(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, EmitCode) => Unit): Unit = {
    val n = cb.newLocal[Long]("bll_foreach_n")
    foreachNode(cb, n) { cb =>
      val i = cb.newLocal[Int]("bll_foreach_i")
      cb.assign(i, 0)
      cb.whileLoop(i < count(n),
        {
          f(cb, EmitCode(Code._empty, bufferType.isElementMissing(buffer(n), i),
            elemType.loadCheapPCode(cb, bufferType.loadElement(buffer(n), capacity(n), i))))
          cb.assign(i, i + 1)
        })
    }
  }

  private def pushImpl(cb: EmitCodeBuilder, r: Value[Region], v: EmitCode): Unit = {
    cb.ifx(count(lastNode) >= capacity(lastNode),
      cb += pushNewBlockNode(cb.emb, r, defaultBlockCap))
    v.toI(cb)
      .consume(cb,
        cb += pushMissing(lastNode),
        { sc =>
          pushPresent(cb, lastNode) { (cb, addr) =>
            elemType.storeAtAddress(cb, addr, r, sc, deepCopy = true)
          }
        })

    cb.assign(totalCount, totalCount + 1)
  }

  def push(cb: EmitCodeBuilder, region: Value[Region], elt: EmitCode): Unit = {
    val pushF = kb.genEmitMethod("blockLinkedListPush",
      FastIndexedSeq[ParamType](typeInfo[Region], elt.emitParamType), typeInfo[Unit])
    pushF.voidWithBuilder { cb =>
      pushImpl(cb,
        pushF.getCodeParam[Region](1),
        pushF.getEmitParam(2, null)) // don't need region
    }
    cb.invokeVoid(pushF, region, elt)
  }

  def append(cb: EmitCodeBuilder, region: Value[Region], bll: StagedBlockLinkedList): Unit = {
    // it would take additional logic to get self-append to work, but we don't need it to anyways
    assert(bll ne this)
    assert(bll.elemType.isOfType(elemType))
    val appF = kb.genEmitMethod("blockLinkedListAppend",
      FastIndexedSeq[ParamType](typeInfo[Region]),
      typeInfo[Unit])
    appF.voidWithBuilder { cb =>
      bll.foreach(cb) { (cb, elt) =>
        pushImpl(cb, appF.getCodeParam[Region](1), elt)
      }
    }
    cb.invokeVoid(appF, region)
  }

  def resultArray(cb: EmitCodeBuilder, region: Value[Region], resType: PCanonicalArray): SIndexablePointerCode = {
    val (pushElement, finish) = resType.constructFromFunctions(cb, region, totalCount, deepCopy = true)
    foreach(cb) { (cb, elt) =>
      pushElement(cb, elt.toI(cb))
    }
    finish(cb)
  }

  def serialize(cb: EmitCodeBuilder, region: Value[Region], outputBuffer: Code[OutputBuffer]): Unit = {
    val serF = kb.genEmitMethod("blockLinkedListSerialize",
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[OutputBuffer]),
      typeInfo[Unit])
    val ob = serF.getCodeParam[OutputBuffer](2)
    serF.voidWithBuilder { cb =>
      val n = cb.newLocal[Long]("bll_serialize_n")
      val i = cb.newLocal[Int]("bll_serialize_i")
      val b = cb.newLocal[Long]("bll_serialize_b")
      foreachNode(cb, n) { cb =>
        cb += ob.writeBoolean(true)
        cb.assign(b, buffer(n))
        bufferEType.buildPrefixEncoder(cb, bufferType.loadCheapPCode(cb, b).memoize(cb, "sbll_serialize_v"), ob, count(n))
      }
      cb += ob.writeBoolean(false)
    }
    cb.invokeVoid(serF, region, outputBuffer)
  }

  def deserialize(cb: EmitCodeBuilder, region: Code[Region], inputBuffer: Code[InputBuffer]): Unit = {
    val desF = kb.genEmitMethod("blockLinkedListDeserialize",
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[InputBuffer]),
      typeInfo[Unit])
    val r = desF.getCodeParam[Region](1)
    val ib = desF.getCodeParam[InputBuffer](2)
    val dec = bufferEType.buildDecoder(bufferType.virtualType, desF.ecb)
    desF.voidWithBuilder { cb =>
      cb.whileLoop(ib.readBoolean(), {
        appendShallow(cb, r, dec(cb, r, ib))
      })
    }
    cb.invokeVoid(desF, region, inputBuffer)
  }

  private def appendShallow(cb: EmitCodeBuilder, r: Code[Region], aCode: PCode): Unit = {
    val buff = cb.memoize(aCode, "sbll_append_shallow_a").asInstanceOf[SIndexablePointerSettable]
    val newNode = cb.newLocal[Long]("sbll_append_shallow_newnode", nodeType.allocate(r))
    cb += initNode(newNode, buf = buff.a, count = buff.length)
    cb += setNext(lastNode, newNode)
    cb.assign(lastNode, newNode)
    cb.assign(totalCount, totalCount + buff.length)
  }

  def initWithDeepCopy(cb: EmitCodeBuilder, region: Value[Region], other: StagedBlockLinkedList): Unit = {
    assert(other ne this)
    assert(other.kb eq kb)
    val initF = kb.genEmitMethod("blockLinkedListDeepCopy",
      FastIndexedSeq[ParamType](typeInfo[Region]),
      typeInfo[Unit])
    val r = initF.getCodeParam[Region](1)
    initF.voidWithBuilder { cb =>
      // sets firstNode
      cb += initWithCapacity(r, other.totalCount)
      val i = cb.newLocal[Int]("sbll_init_deepcopy_i")
      val buf = cb.newLocal[Long]("sbll_init_deepcopy_buf", buffer(firstNode))
      other.foreach(cb) { (cb, elt) =>
        elt.toI(cb)
          .consume(cb,
            cb += bufferType.setElementMissing(buf, i),
            { sc =>
              cb += bufferType.setElementPresent(buf, i)
              elemType.storeAtAddress(cb, bufferType.elementOffset(buf, i), r, sc, deepCopy = true)
            })
        cb += incrCount(firstNode)
        cb.assign(i, i + 1)
      }
      cb.assign(totalCount, other.totalCount)
    }
    cb.invokeVoid(initF, region)
  }
}
