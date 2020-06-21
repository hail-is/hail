package is.hail.lir

import java.util.Comparator

import is.hail.asm4s.{BooleanInfo, IntInfo, TypeInfo, UnitInfo}
import is.hail.utils.{ArrayBuilder, FastIndexedSeq, UnionFind}
import org.objectweb.asm.Opcodes._

object SplitMethod {
  val TargetMethodSize: Int = 2000

  def apply(
    c: Classx[_],
    m: Method,
    blocks: IndexedSeq[Block],
    pst: PST
  ): Classx[_] = {
    val split = new SplitMethod(c, m, blocks, pst)
    split.split()
    split.spillsClass
  }
}

class SplitMethod(
  c: Classx[_],
  m: Method,
  blocks: IndexedSeq[Block],
  pst: PST
) {
  def nBlocks: Int = blocks.length

  val blockPartitions = new UnionFind(nBlocks)
  (0 until nBlocks).foreach { i => blockPartitions.makeSet(i) }

  private var methodSize = 0
  private val regionSize = new Array[Int](pst.nRegions)

  // compute methodSize and regionSize
  private def computeSizes(): Unit = {
    val blockSize = new Array[Int](nBlocks)

    var i = 0
    while (i < nBlocks) {
      val size = blocks(i).approxByteCodeSize()
      blockSize(i) = size
      methodSize += size
      i += 1
    }

    i = pst.nRegions - 1
    while (i >= 0) {
      val r = pst.regions(i)

      var c = 0
      var ci = 0
      var child: Region = null

      if (c < r.children.length) {
        ci = r.children(c)
        assert(ci > i)
        child = pst.regions(ci)
      }

      var size = 0
      var j = r.start
      while (j <= r.end) {
        if (child != null && j == child.start) {
          // children can overlap, don't double-count
          size += regionSize(ci) - blockSize(child.end)
          j = child.end
          c += 1
          if (c < r.children.length) {
            ci = r.children(c)
            assert(ci > i)
            child = pst.regions(ci)
          }
        } else {
          assert(child == null || j < child.start)
          size += blockSize(j)
          j += 1
        }
      }
      regionSize(i) = size

      i -= 1
    }
  }

  private val spillsClass = new Classx(genName("C", s"${ m.name }Spills"), "java/lang/Object")
  private val spillsCtor = {
    val ctor = spillsClass.newMethod("<init>", FastIndexedSeq(), UnitInfo)
    val L = new Block()
    ctor.setEntry(L)
    L.append(
      methodStmt(INVOKESPECIAL,
        "java/lang/Object",
        "<init>",
        "()V",
        false,
        UnitInfo,
        FastIndexedSeq(load(ctor.getParam(0)))))
    L.append(returnx())
    ctor
  }

  private val spills = m.newLocal("spills", spillsClass.ti)

  private val paramFields = m.parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
    spillsClass.newField(genName("f", s"arg${ i + 1 }"), ti)
  }

  def splitLargeStatements(): Unit = {
    for (b <- blocks) {
      def splitLargeStatement(c: StmtX): Unit = {
        def visit(x: ValueX): Int = {
          // FIXME this doesn't handle many moderate-sized children
          val size = 1 + x.children.map(visit).sum

          if (size > SplitMethod.TargetMethodSize / 2) {
            val l = m.newLocal("spill_large_expr", x.ti)
            x.replace(load(l))
            c.insertBefore(store(l, x))
            1
          } else
            size
        }

        c.children.foreach(visit)
      }

      var x = b.first
      while (x != null) {
        splitLargeStatement(x)
        x = x.next
      }
    }
  }

  def spillLocals(): Unit = {
    val locals = m.findLocals(blocks)

    val fields = locals.map { l =>
      if (l.isInstanceOf[Parameter])
        null
      else
        spillsClass.newField(genName("f", l.name), l.ti)
    }

    def localField(l: Local): Field =
      l match {
        case p: Parameter =>
          if (p.i == 0)
            null
          else
            paramFields(p.i - 1)
        case _ => fields(locals.index(l))
      }

    def spillsParam(): Parameter =
      new Parameter(null, 1, spillsClass.ti)

    def spill(x: X): Unit = {
      x.children.foreach(spill)
      x match {
        case x: LoadX =>
          val f = localField(x.l)
          if (f != null)
            x.replace(getField(f, load(spillsParam())))
        case x: IincX =>
          val f = localField(x.l)
          assert(f != null)
          x.replace(
            putField(f, load(spillsParam()),
              insn(IADD,
                getField(f, load(spillsParam())),
                ldcInsn(x.i, IntInfo))))
        case x: StoreX =>
          val f = localField(x.l)
          assert(f != null)
          val v = x.children(0)
          v.remove()
          x.replace(putField(f, load(spillsParam()), v))
        case _ =>
      }
    }

    for (b <- blocks) {
      var x = b.first
      while (x != null) {
        val n = x.next
        spill(x)
        x = n
      }
    }
  }

  def splitBlock(b: Block): Unit = {
    val last = b.last

    val returnTI = last match {
      case _: GotoX => UnitInfo
      case _: IfX => BooleanInfo
      case _: SwitchX => IntInfo
      case _: ReturnX => m.returnTypeInfo
      case _: ThrowX => last.children(0).ti
    }

    var L = new Block()
    var x = b.first
    var size = 0

    while (x != last) {
      if (size > SplitMethod.TargetMethodSize) {
        val newM = c.newMethod(genName("m", "wrapped"), FastIndexedSeq(spillsClass.ti), UnitInfo)
        L.method = newM
        newM.setEntry(L)
        L.append(returnx())

        x.insertBefore(methodStmt(INVOKEVIRTUAL, newM, Array(load(m.getParam(0)), load(spills))))

        L = new Block()
        size = 0
      }

      size += x.approxByteCodeSize()
      val n = x.next
      x.remove()
      L.append(x)
      x = n
    }

    val newM = c.newMethod(genName("m", "wrapped"), FastIndexedSeq(spillsClass.ti), returnTI)
    L.method = newM
    newM.setEntry(L)

    def invokeNewM(): ValueX =
      methodInsn(INVOKEVIRTUAL, newM, Array(load(m.getParam(0)), load(spills)))

    def invokeNewMStmt(): StmtX =
      methodStmt(INVOKEVIRTUAL, newM, Array(load(m.getParam(0)), load(spills)))

    last match {
      case _: GotoX =>
        L.append(returnx())
        last.insertBefore(invokeNewMStmt())
      case x: ThrowX =>
        val err = x.children(0)
        x.setChild(0, invokeNewM())
        L.append(returnx(err))
      case x: IfX =>
        val Ltrue = x.Ltrue
        val Lfalse = x.Lfalse

        x.remove()
        L.append(x)

        val newLtrue = new Block()
        newLtrue.method = newM
        newLtrue.append(returnx(ldcInsn(1, BooleanInfo)))
        x.setLtrue(newLtrue)

        val newLfalse = new Block()
        newLfalse.method = newM
        newLfalse.append(returnx(ldcInsn(0, BooleanInfo)))
        x.setLfalse(newLfalse)

        b.append(
          ifx(IFNE, invokeNewM(), Ltrue, Lfalse))
      case x: SwitchX => IntInfo
        val i = x.children(0)
        x.setChild(0, invokeNewM())
        L.append(returnx(i))
      case _: ReturnX =>
        if (returnTI eq UnitInfo) {
          L.append(returnx())
          last.insertBefore(invokeNewMStmt())
        } else {
          val c = x.children(0)
          x.setChild(0, invokeNewM())
          L.append(returnx(c))
        }
    }
  }

  val q = new java.util.TreeSet[Integer](
    new Comparator[Integer] {
      def compare(i: Integer, j: Integer): Int =
        Integer.compare(regionSize(i), regionSize(j))
    })

  private val splitMethods = new ArrayBuilder[Method]()

  private var counter = 0
  private def genSplitMethodName(): String = {
    val c = counter
    counter += 1
    s"${ m.name }split$c"
  }

  private var splitReturn: Field = null
  private var splitReturnValue: Field = null

  private def splitRegion(i: Int): Unit = {
    val r = pst.regions(i)

    // don't include end
    val splittingReturn = (r.start until r.end).exists { i =>
      val L = blocks(blockPartitions.find(pst.linearization(i)))
      L.last.isInstanceOf[ReturnX]
    }

    if (splittingReturn && splitReturn == null) {
      splitReturn = spillsClass.newField(genName("f", "splitReturn"), BooleanInfo)
      if (m.returnTypeInfo != UnitInfo) {
        splitReturnValue = spillsClass.newField(genName("f", "splitReturnValue"), m.returnTypeInfo)
      }
    }

    val Lstart = blocks(pst.linearization(r.start))
    val Lend = blocks(pst.linearization(r.end))

    val returnTI = Lend.last match {
      case _: GotoX => UnitInfo
      case _: IfX => BooleanInfo
      case _: SwitchX => IntInfo
      case _: ReturnX => m.returnTypeInfo
      case _: ThrowX => UnitInfo
    }

    val splitM = c.newMethod(genSplitMethodName(), FastIndexedSeq(spillsClass.ti), returnTI)

    (r.start to r.end).foreach { i =>
      val b = blockPartitions.find(pst.linearization(i))
      val L = blocks(b)
      blocks(b) = null
      L.method = splitM
    }

    splitM.setEntry(Lstart)

    val L = new Block()
    L.method = m

    blockPartitions.union(
      pst.linearization(r.start), pst.linearization(r.end))
    blocks(blockPartitions.find(pst.linearization(r.start))) = L

    def invokeNewM(): ValueX =
      methodInsn(INVOKEVIRTUAL, splitM, Array(load(m.getParam(0)), load(spills)))

    def invokeNewMStmt(): StmtX =
      methodStmt(INVOKEVIRTUAL, splitM, Array(load(m.getParam(0)), load(spills)))

    Lend.last match {
      case x: GotoX =>
        x.remove()
        Lend.append(returnx())
        L.append(invokeNewMStmt())
        L.append(x)
      case x: ThrowX =>
        L.append(invokeNewMStmt())
      case x: IfX =>
        val Ltrue = x.Ltrue
        val Lfalse = x.Lfalse

        val newLtrue = new Block()
        newLtrue.method = splitM
        newLtrue.append(returnx(ldcInsn(1, BooleanInfo)))
        x.Ltrue = newLtrue

        val newLfalse = new Block()
        newLfalse.method = splitM
        newLfalse.append(returnx(ldcInsn(0, BooleanInfo)))
        x.Lfalse = newLfalse

        L.append(
          ifx(IFNE, invokeNewM(), Ltrue, Lfalse))
      case x: SwitchX => IntInfo
        val i = x.children(0)
        x.setChild(0, invokeNewM())
        Lend.append(returnx(i))
      case x: ReturnX =>
        x.remove()

        if (returnTI eq UnitInfo) {
          L.append(returnx())
          last.insertBefore(invokeNewMStmt())
        } else {
          val c = x.children(0)
          x.setChild(0, invokeNewM())
          L.append(returnx(c))
        }
    }


    ???

    // update sizes
    val size = regionSize(i)

    methodSize -= size

    val p = pst.regions(i).parent
    while (p != -1) {
      q.remove(p)
      regionSize(p) -= size
      assert(pst.regions(p).children.nonEmpty)
      if (regionSize(p) < SplitMethod.TargetMethodSize) {
        q.add(p)
      }
    }
  }

  def splitRegions(): Unit = {
    var i = 0
    while (i < pst.nRegions) {
      if (regionSize(i) < SplitMethod.TargetMethodSize ||
        pst.regions(i).children.isEmpty) {
        q.add(i)
      }
      i += 1
    }

    while (methodSize > SplitMethod.TargetMethodSize &&
      !q.isEmpty) {
      val i = q.pollLast()
      splitRegion(i)
    }
  }

  def split(): Unit = {
    computeSizes()
    splitRegions()

    splitLargeStatements()
    spillLocals()

    for (b <- blocks) {
      splitBlock(b)
    }

    val newEntry = new Block()
    newEntry.method = m
    newEntry.append(store(spills, new NewInstanceX(spillsClass.ti, spillsCtor)))

    // this can't get split
    m.parameterTypeInfo.indices.foreach { i =>
      newEntry.append(putField(
        paramFields(i),
        load(spills),
        load(m.getParam(i + 1))))
    }
    newEntry.append(goto(m.entry))
    m.setEntry(newEntry)
  }
}
