package is.hail.lir

import is.hail.asm4s._
import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm._

import scala.collection.mutable

class SplitReturn() extends Exception()

class SplitUnreachable() extends Exception()

object SplitMethod {
  // FIXME
  val TargetMethodSize: Int = 50

  def apply(
    c: Classx[_],
    m: Method,
    pst: PST
  ): Classx[_] = {
    val split = new SplitMethod(c, m, pst.blocks, pst)
    split.split()
    split.spillsClass
  }
}

class SplitMethod(
  c: Classx[_],
  m: Method,
  blocks: Blocks,
  pst: PST
) {
  def nBlocks: Int = blocks.nBlocks

  private val blockPartitions = new UnionFind(nBlocks)
  (0 until nBlocks).foreach { i => blockPartitions.makeSet(i) }

  private var methodSize = 0

  private val blockSize = new Array[Int](nBlocks)

  // compute methodSize and regionSize
  private def computeBlockSizes(): Unit = {
    var i = 0
    while (i < nBlocks) {
      val size = blocks(i).approxByteCodeSize()
      blockSize(i) = size
      methodSize += size
      i += 1
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

  private val spillReturnValue: Field = {
    if (m.returnTypeInfo != UnitInfo)
      spillsClass.newField(genName("f", "spillReturnValue"), m.returnTypeInfo)
    else
      null
  }

  def throwUnreachable(): ControlX = {
    val ti = classInfo[SplitUnreachable]
    val tcls = classOf[SplitUnreachable]
    val c = tcls.getDeclaredConstructor()
    throwx(newInstance(ti,
      Type.getInternalName(tcls), "<init>", Type.getConstructorDescriptor(c), ti, FastIndexedSeq()))
  }

  // index before creating spills
  private val locals = m.findLocals(blocks)

  private val spills = m.newLocal("spills", spillsClass.ti)

  private val paramFields = m.parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
    spillsClass.newField(genName("f", s"arg${ i + 1 }"), ti)
  }

  private val fields = locals.map { l =>
    if (l.isInstanceOf[Parameter])
      null
    else
      spillsClass.newField(genName("f", l.name), l.ti)
  }

  private val splitMethods = mutable.ArrayBuffer[Method]()

  // also fixes up null switch targets
  def spillLocals(method: Method): Unit = {
    def localField(l: Local): Field =
      l match {
        case p: Parameter =>
          if (p.i == 0)
            null
          else {
            if (method eq m)
              null
            else
              paramFields(p.i - 1)
          }
        case _ =>
          locals.getIndex(l) match {
            case Some(i) =>
              fields(i)
            case None =>
              null
          }
      }

    def getSpills(): ValueX = {
      if (method eq m)
        load(spills)
      else
        load(new Parameter(method, 1, spillsClass.ti))
    }

    def spill(x: X): Unit = {
      x.children.foreach(spill)
      x match {
        case x: LoadX =>
          if ((method ne m) && (x.l eq spills)) {
            x.l = new Parameter(method, 1, spillsClass.ti)
          } else {
            val f = localField(x.l)
            if (f != null)
              x.replace(getField(f, getSpills()))
          }
        case x: IincX =>
          assert(x.l ne spills)
          val f = localField(x.l)
          if (f != null) {
            x.replace(
              putField(f, getSpills(),
                insn(IADD,
                  getField(f, getSpills()),
                  ldcInsn(x.i, IntInfo))))
          }
        case x: StoreX =>
          assert(x.l ne spills)
          val f = localField(x.l)
          if (f != null) {
            val v = x.children(0)
            v.remove()
            x.replace(putField(f, getSpills(), v))
          }
        case _ =>
      }
    }

    val blocks = method.findBlocks()
    for (b <- blocks) {
      b.last match {
        case x: SwitchX =>
          var Lunreachable: Block = null
          var i = 0
          while (i < x.targetArity()) {
            if (x.target(i) == null) {
              if (Lunreachable == null) {
                Lunreachable = new Block()
                Lunreachable.method = method
                Lunreachable.append(throwUnreachable())
              }
              x.setTarget(i, Lunreachable)
            }
            i += 1
          }
        case _ =>
      }

      var x = b.first
      while (x != null) {
        val n = x.next
        spill(x)
        x = n
      }
    }
  }

  def spillLocals(): Unit = {
    for (splitM <- splitMethods) {
      spillLocals(splitM)
    }
    spillLocals(m)
  }

  // updated while we split
  private val updatedBlocks = blocks.toArray

  private def splitSlice(start: Int, end: Int): Unit = {
    val Lstart = updatedBlocks(blockPartitions.find(start))
    val Lend = updatedBlocks(blockPartitions.find(end))

    val regionBlocks = mutable.Set[Block]()
    (start to end).foreach { i =>
      val b = blockPartitions.find(i)
      val L = updatedBlocks(b)
      if (L != null) {
        updatedBlocks(b) = null
        regionBlocks += L
      }
    }

    // replacement block for region
    val newL = new Block()
    newL.method = m

    // don't iterate over set that's being modified
    val uses2 = Lstart.uses.toArray
    for ((x, i) <- uses2) {
      val xL = x.containingBlock()
      assert(xL != null)
      if (!regionBlocks(xL))
        x.setTarget(i, newL)
    }
    if (m.entry == Lstart)
      m.setEntry(newL)

    (start to end).foreach { i =>
      blockPartitions.union(start, i)
    }
    updatedBlocks(blockPartitions.find(start)) = newL

    val returnTI = Lend.last match {
      case _: GotoX => UnitInfo
      case x: IfX =>
        if (!regionBlocks(x.Ltrue) && !regionBlocks(x.Lfalse))
          BooleanInfo
        else
          UnitInfo
      case _: SwitchX => IntInfo
      case _: ReturnX => m.returnTypeInfo
      case _: ThrowX => UnitInfo
    }

    val splitM = c.newMethod(s"${ m.name }region${ start }_$end", FastIndexedSeq(spillsClass.ti), returnTI)
    splitMethods += splitM

    splitM.setEntry(Lstart)

    for (b <- regionBlocks) {
      b.method = splitM

      // handle split return statements
      if (b ne Lend) {
        b.last match {
          case x: ReturnX =>
            x.remove()
            if (m.returnTypeInfo != UnitInfo) {
              val v = x.children(0)
              v.remove()
              b.append(putField(spillReturnValue, load(spills), v))
            }
            val ti = classInfo[SplitReturn]
            val tcls = classOf[SplitReturn]
            val c = tcls.getDeclaredConstructor()
            b.append(
              throwx(newInstance(ti,
                Type.getInternalName(tcls), "<init>", Type.getConstructorDescriptor(c), ti, FastIndexedSeq())))
          case _ =>
        }
      }
    }

    def invokeSplitM(): ValueX =
      methodInsn(INVOKEVIRTUAL, splitM, Array(load(m.getParam(0)), load(spills)))

    def invokeSplitMStmt(): StmtX =
      methodStmt(INVOKEVIRTUAL, splitM, Array(load(m.getParam(0)), load(spills)))

    Lend.last match {
      case x: GotoX =>
        newL.append(invokeSplitMStmt())
        if (regionBlocks(x.L)) {
          val ti = classInfo[SplitUnreachable]
          val tcls = classOf[SplitUnreachable]
          val c = tcls.getDeclaredConstructor()
          newL.append(
            throwx(newInstance(ti,
              Type.getInternalName(tcls), "<init>", Type.getConstructorDescriptor(c), ti, FastIndexedSeq())))
        } else {
          x.remove()
          Lend.append(returnx())
          newL.append(x)
        }
      case _: ThrowX =>
        newL.append(invokeSplitMStmt())
        newL.append(throwUnreachable())
      case x: IfX =>
        if (regionBlocks(x.Ltrue)) {
          if (regionBlocks(x.Lfalse)) {
            newL.append(invokeSplitMStmt())
            newL.append(throwUnreachable())
          } else {
            newL.append(invokeSplitMStmt())
            newL.append(goto(x.Lfalse))
            val Lreturn = new Block()
            Lreturn.method = splitM
            Lreturn.append(returnx())
            x.setLfalse(Lreturn)
          }
        } else {
          if (regionBlocks(x.Lfalse)) {
            newL.append(invokeSplitMStmt())
            newL.append(goto(x.Ltrue))
            val Lreturn = new Block()
            Lreturn.method = splitM
            Lreturn.append(returnx())
            x.setLtrue(Lreturn)
          } else {
            newL.append(
              ifx(IFNE, invokeSplitM(), x.Ltrue, x.Lfalse))

            val newLtrue = new Block()
            newLtrue.method = splitM
            newLtrue.append(returnx(ldcInsn(1, BooleanInfo)))
            x.setLtrue(newLtrue)

            val newLfalse = new Block()
            newLfalse.method = splitM
            newLfalse.append(returnx(ldcInsn(0, BooleanInfo)))
            x.setLfalse(newLfalse)
          }
        }
      case x: SwitchX => IntInfo
        // FIXME potential for optimization like if
        val idx = x.children(0)
        idx.remove()
        val lidx = splitM.newLocal("switch_index", IntInfo)
        x.insertBefore(store(lidx, idx))
        x.setChild(0, load(lidx))
        val Lreturn = new Block()
        Lreturn.method = splitM
        Lreturn.append(returnx(load(lidx)))
        val newSwitch = switch(invokeSplitM(), x.Ldefault, x.Lcases)
        var i = 0
        while (i < x.targetArity()) {
          val L = x.target(i)
          if (regionBlocks(L))
            // null is replaced with throw new SplitUnreachable
            newSwitch.setTarget(i, null)
          else
            x.setTarget(i, Lreturn)
          i += 1
        }
        newL.append(newSwitch)
      case _: ReturnX =>
        if (m.returnTypeInfo == UnitInfo) {
          newL.append(invokeSplitMStmt())
          newL.append(returnx())
        } else {
          newL.append(returnx(invokeSplitM()))
        }
    }

    blockSize(blockPartitions.find(start)) = newL.approxByteCodeSize()
  }

  // utility  class
  class SizedRange(val start: Int, val end: Int, val size: Int) {
    override def toString: String = s"$start-$end/$size"
  }

  def splitSubregions(subr0: mutable.ArrayBuffer[SizedRange]): Int = {
    var subr = subr0

    var size = subr.iterator.map(_.size).sum

    println(s"subr0 $size")
    println(subr)

    var changed = true
    while (changed &&
      size > SplitMethod.TargetMethodSize) {
      println(s"subr $size")
      println(subr)

      changed = false

      val coalescedsubr = new mutable.ArrayBuffer[SizedRange]()

      var i = 0
      while (i < subr.size) {
        var s = subr(i).size
        var j = i + 1
        while (j < subr.size &&
          subr(i).end == subr(j).start &&
          (s + subr(j).size < SplitMethod.TargetMethodSize)) {
          s += subr(j).size
          j += 1
        }
        coalescedsubr += new SizedRange(subr(i).start, subr(j - 1).end, s)
        i = j
      }

      val sortedsubr = coalescedsubr.sortBy(_.size)

      val newsubr = mutable.ArrayBuffer[SizedRange]()

      i = sortedsubr.length - 1
      while (i >= 0) {
        val ri = sortedsubr(i)
        if (ri.size > 20 &&
          size > SplitMethod.TargetMethodSize) {

          size -= ri.size
          splitSlice(ri.start, ri.end)
          val s = blockSize(blockPartitions.find(ri.start))
          assert(s < 20)
          size += s

          newsubr += new SizedRange(ri.start, ri.end, s)

          changed = true
        } else
          newsubr += ri

        i -= 1
      }

      subr = newsubr.sortBy(_.start)
    }

    size
  }

  def splitRegions(): Unit = {
    val regionSize = new Array[Int](pst.nRegions)
    var i = 0
    while (i < pst.nRegions) {
      val r = pst.regions(i)
      if (r.children.nonEmpty) {
        val subr = mutable.ArrayBuffer[SizedRange]()
        subr ++= r.children.iterator.map { ci =>
          val cr = pst.regions(ci)
          assert(ci < i)
          new SizedRange(cr.start, cr.end, regionSize(ci))
        }
        regionSize(i) = splitSubregions(subr)
      } else {
        assert(r.start == r.end)
        regionSize(i) = blockSize(blockPartitions.find(r.start))
      }
      i += 1
    }
  }

  def split(): Unit = {
    computeBlockSizes()
    splitRegions()

    spillLocals()

    m.spillsInit = store(spills, new NewInstanceX(spillsClass.ti, spillsCtor))

    if (m.returnTypeInfo != UnitInfo)
      m.spillsReturn = returnx(getField(spillReturnValue, load(spills)))
    else {
      m.spillsReturn = returnx()
    }

    // spill parameters
    var x: StmtX = null
    m.parameterTypeInfo.indices.foreach { i =>
      val putParam = putField(
        paramFields(i),
        load(spills),
        load(m.getParam(i + 1)))
      if (x != null)
        x.insertAfter(putParam)
      else
        m.entry.prepend(putParam)
      x = putParam
    }
  }
}
