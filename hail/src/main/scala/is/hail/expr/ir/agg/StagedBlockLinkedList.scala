package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

object StagedBlockLinkedList {
  val defaultBlockCap: Int = 64

  private val nil: Long = 0L
}

class StagedBlockLinkedList(val elemType: PType, val cb: EmitClassBuilder[_]) {
  import StagedBlockLinkedList._

  private val firstNode = cb.genFieldThisRef[Long]()
  private val lastNode = cb.genFieldThisRef[Long]()
  private val totalCount = cb.genFieldThisRef[Int]()

  val storageType = PCanonicalStruct(
    "firstNode" -> PInt64Required,
    "lastNode" -> PInt64Required,
    "totalCount" -> PInt32Required)

  def load(src: Code[Long]): Code[Unit] =
    Code.memoize(src, "sbll_load_src") { src =>
      Code(
        firstNode := Region.loadAddress(storageType.fieldOffset(src, "firstNode")),
        lastNode := Region.loadAddress(storageType.fieldOffset(src, "lastNode")),
        totalCount := Region.loadInt(storageType.fieldOffset(src, "totalCount")))
    }

  def store(dst: Code[Long]): Code[Unit] =
    Code.memoize(dst, "sbll_store_dst") { dst =>
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
    Code.memoize(n, "sbll_init_node_n") { n =>
      Code(
        Region.storeAddress(nodeType.fieldOffset(n, "buf"), buf),
        Region.storeInt(nodeType.fieldOffset(n, "count"), count),
        Region.storeAddress(nodeType.fieldOffset(n, "next"), nil))
    }

  private def pushPresent(n: Node, store: Code[Long] => Code[Unit]): Code[Unit] =
    Code(
      if (elemType.required) Code._empty else bufferType.setElementPresent(buffer(n), count(n)),
      store(bufferType.elementOffset(buffer(n), capacity(n), count(n))),
      incrCount(n))

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

  def init(r: Value[Region]): Code[Unit] =
    initWithCapacity(r, defaultBlockCap)

  private def pushNewBlockNode(mb: EmitMethodBuilder[_], r: Value[Region], cap: Code[Int]): Code[Unit] = {
    val newNode = mb.newLocal[Long]()
    Code(
      allocateNode(newNode)(r, cap),
      setNext(lastNode, newNode),
      lastNode := newNode)
  }

  private def foreachNode(mb: EmitMethodBuilder[_], tmpNode: Settable[Long])(body: Code[Unit]): Code[Unit] = {
    val present = mb.newLocal[Boolean]()
    Code(
      tmpNode := firstNode,
      present := true,
      Code.whileLoop(present,
        body,
        present := hasNext(tmpNode),
        tmpNode := next(tmpNode)))
  }

  private def foreach(mb: EmitMethodBuilder[_])(f: EmitCode => Code[Unit]): Code[Unit] = {
    val n = mb.newLocal[Long]()
    foreachNode(mb, n) {
      val i = mb.newLocal[Int]()
      val bufim = bufferType.isElementMissing(buffer(n), i)
      val bufiv = Region.loadIRIntermediate(elemType)(bufferType.elementOffset(buffer(n), capacity(n), i))
      Code(
        i := 0,
        Code.whileLoop(i < count(n),
          f(EmitCode(Code._empty, bufim, PCode(elemType, bufiv))),
          i := i + 1))
    }
  }

  private def push(mb: EmitMethodBuilder[_], r: Value[Region], m: Code[Boolean], v: Code[_]): Code[Unit] = {
    var push = pushPresent(lastNode, StagedRegionValueBuilder.deepCopy(cb, r, elemType, v, _))
    if (!elemType.required)
      push = m.mux(pushMissing(lastNode), push)
    Code(
      (count(lastNode) >= capacity(lastNode)).orEmpty(
        pushNewBlockNode(mb, r, defaultBlockCap)), // push a new block if lastNode is full
      push,
      totalCount := totalCount + 1)
  }

  def push(region: Value[Region], elt: EmitCode): Code[Unit] = {
    val eltTI = typeToTypeInfo(elemType)
    val pushF = cb.genEmitMethod("blockLinkedListPush",
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[Boolean], eltTI),
      typeInfo[Unit])
    pushF.emit(push(pushF,
      pushF.getCodeParam[Region](1),
      pushF.getCodeParam[Boolean](2),
      pushF.getCodeParam(3)(eltTI).get))
    Code(
      elt.setup,
      elt.m.mux(
        pushF.invokeCode(region, const(true), defaultValue(elemType)),
        pushF.invokeCode(region, const(false), elt.v)))
  }

  def append(region: Value[Region], bll: StagedBlockLinkedList): Code[Unit] = {
    // it would take additional logic to get self-append to work, but we don't need it to anyways
    assert(bll ne this)
    assert(bll.elemType.isOfType(elemType))
    val appF = cb.genEmitMethod("blockLinkedListAppend",
      FastIndexedSeq[ParamType](typeInfo[Region]),
      typeInfo[Unit])
    appF.emit(bll.foreach(appF) { elt =>
      push(appF, appF.getCodeParam[Region](1), elt.m, elt.v)
    })
    appF.invokeCode(region)
  }

  def writeToSRVB(mb: EmitMethodBuilder[_], srvb: StagedRegionValueBuilder): Code[Unit] = {
    assert(srvb.typ.fundamentalType.isOfType(bufferType.fundamentalType), s"srvb: ${srvb.typ}, buf: ${bufferType.fundamentalType}")
      Code(
        srvb.start(totalCount, init = true),
        foreach(mb) { elt =>
          Code(
            elt.setup,
            elt.m.mux(
              srvb.setMissing(),
              srvb.addWithDeepCopy(elemType, elt.value)),
            srvb.advance())
        })
  }

  def serialize(region: Code[Region], outputBuffer: Code[OutputBuffer]): Code[Unit] = {
    val serF = cb.genEmitMethod("blockLinkedListSerialize",
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[OutputBuffer]),
      typeInfo[Unit])
    val ob = serF.getCodeParam[OutputBuffer](2)
    serF.emit {
      val n = serF.newLocal[Long]()
      val i = serF.newLocal[Int]()
      val b = serF.newLocal[Long]()
      Code(
        foreachNode(serF, n) { Code(
          ob.writeBoolean(true),
          b := buffer(n),
          bufferEType.buildPrefixEncoder(bufferType.encodableType, serF, b, ob, count(n)))
        },
        ob.writeBoolean(false))
    }
    serF.invokeCode(region, outputBuffer)
  }

  def deserialize(region: Code[Region], inputBuffer: Code[InputBuffer]): Code[Unit] = {
    val desF = cb.genEmitMethod("blockLinkedListDeserialize",
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[InputBuffer]),
      typeInfo[Unit])
    val r = desF.getCodeParam[Region](1)
    val ib = desF.getCodeParam[InputBuffer](2)
    val array = desF.newLocal[Long]("array")
    val dec = bufferEType.buildDecoder(bufferType, desF.ecb)
    desF.emit(
      Code.whileLoop(ib.readBoolean(),
        array := dec(r, ib),
        appendShallow(desF, r, array))
    )
    desF.invokeCode[Unit](region, inputBuffer)
  }

  private def appendShallow(mb: EmitMethodBuilder[_], r: Code[Region], aoff: Code[Long]): Code[Unit] = {
    Code.memoize(aoff, "sbll_append_shallow_aoff") { aoff =>
      Code.memoize(bufferType.loadLength(aoff), "sbll_append_shallow_len") { len =>
        val newNode = mb.newLocal[Long]()
        Code(
          newNode := r.allocate(nodeType.alignment, nodeType.byteSize),
          initNode(newNode,
            buf = aoff,
            count = len),
          setNext(lastNode, newNode),
          lastNode := newNode,
          totalCount := totalCount + len)
      }
    }
  }

  def initWithDeepCopy(region: Value[Region], other: StagedBlockLinkedList): Code[Unit] = {
    assert(other ne this)
    assert(other.cb eq cb)
    val initF = cb.genEmitMethod("blockLinkedListDeepCopy",
      FastIndexedSeq[ParamType](typeInfo[Region]),
      typeInfo[Unit])
    val r = initF.getCodeParam[Region](1)
    initF.emit {
      val i = initF.newLocal[Int]()
      Code(
        // sets firstNode
        initWithCapacity(r, other.totalCount),
        Code.memoize(buffer(firstNode), "sbll_init_deepcopy_buf") { buf =>
          Code(
            i := 0,
            other.foreach(initF) { et =>
              Code(
                et.setup,
                et.m.mux(bufferType.setElementMissing(buf, i),
                  Code(
                    bufferType.setElementPresent(buf, i),
                    StagedRegionValueBuilder.deepCopy(cb, r, elemType, et.value, bufferType.elementOffset(buf, i)))),
                incrCount(firstNode),
                i += 1)
            },
            totalCount := other.totalCount)
        })
    }
    initF.invokeCode(region)
  }
}
