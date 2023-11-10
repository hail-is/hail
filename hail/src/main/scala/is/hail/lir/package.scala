package is.hail

import is.hail.asm4s.{ArrayInfo, BooleanInfo, ClassInfo, DoubleInfo, FloatInfo, IntInfo, LongInfo, TypeInfo}
import is.hail.utils.FastSeq
import org.objectweb.asm.Opcodes._

package object lir {
  private[this] var counter: Long = 0

  def genName(tag: String, baseName: String): String = synchronized {
    counter += 1
    if (baseName != null) {
      if (baseName.contains("."))
        throw new RuntimeException(s"genName has invalid character(s): $baseName")
      s"__$tag$counter$baseName"
    } else
      s"__$tag${ counter }null"
  }

  def setChildren(x: X, cs: IndexedSeq[ValueX]): Unit = {
    x.setArity(cs.length)

    var i = 0
    while (i < cs.length) {
      x.setChild(i, cs(i))
      i += 1
    }
  }

  def setChildren(x: X, c: ValueX): Unit = {
    x.setArity(1)
    x.setChild(0, c)
  }

  def setChildren(x: X, c1: ValueX, c2: ValueX): Unit = {
    x.setArity(2)
    x.setChild(0, c1)
    x.setChild(1, c2)
  }

  def ifx(op: Int, c: ValueX, Ltrue: Block, Lfalse: Block): ControlX = {
    val x = new IfX(op)
    setChildren(x, c)
    x.setLtrue(Ltrue)
    x.setLfalse(Lfalse)
    x
  }

  def ifx(op: Int, c1: ValueX, c2: ValueX, Ltrue: Block, Lfalse: Block): ControlX = {
    val x = new IfX(op)
    setChildren(x, c1, c2)
    x.setLtrue(Ltrue)
    x.setLfalse(Lfalse)
    x
  }

  def switch(c: ValueX, Ldefault: Block, cases: IndexedSeq[Block]): ControlX = {
    if (cases.isEmpty)
      return goto(Ldefault)

    val x = new SwitchX()
    setChildren(x, c)
    x.setLdefault(Ldefault)
    x.setLcases(cases)
    x
  }

  def goto(L: Block): ControlX = {
    assert(L != null)
    val x = new GotoX
    x.setArity(0)
    x.setL(L)
    x
  }

  def store(l: Local, c: ValueX, lineNumber: Int = 0): StmtX = {
    val x = new StoreX(l, lineNumber)
    setChildren(x, c)
    x
  }

  def iincInsn(l: Local, i: Int): StmtX = new IincX(l, i)

  def insn0(op: Int): ValueX = insn(op, null, FastSeq.empty)

  def insn0(op: Int, _ti: TypeInfo[_]): ValueX = insn(op, _ti, FastSeq.empty)

  def insn1(op: Int): (ValueX) => ValueX = (c) =>
    insn(op, null, FastSeq(c))

  def insn1(op: Int, _ti: TypeInfo[_], lineNumber: Int = 0): (ValueX) => ValueX = (c) =>
    insn(op, _ti, FastSeq(c), lineNumber)

  def insn2(op: Int): (ValueX, ValueX) => ValueX = (c1, c2) =>
    insn(op, null, FastSeq(c1, c2))

  def insn3(op: Int): (ValueX, ValueX, ValueX) => ValueX = (c1, c2, c3) =>
    insn(op, null, FastSeq(c1, c2, c3))

  def insn(op: Int, _ti: TypeInfo[_], args: IndexedSeq[ValueX], lineNumber: Int = 0): ValueX = {
    val x = new InsnX(op, _ti, lineNumber)
    setChildren(x, args)
    x
  }

  def stmtOp3(op: Int): (ValueX, ValueX, ValueX) => StmtX = (c1, c2, c3) => stmtOp(op, c1, c2, c3)

  def stmtOp(op: Int, args: IndexedSeq[ValueX]): StmtX = {
    val x = new StmtOpX(op)
    setChildren(x, args)
    x
  }

  def throwx(c: ValueX, lineNumber: Int = 0): ControlX = {
    val x = new ThrowX(lineNumber)
    setChildren(x, c)
    x
  }

  def stmtOp(op: Int, c1: ValueX, c2: ValueX, c3: ValueX): StmtX = stmtOp(op, FastSeq(c1, c2, c3))

  def load(l: Local): ValueX = new LoadX(l)

  def typeInsn1(op: Int, tti: TypeInfo[_]): (ValueX) => ValueX = (c) => typeInsn(op, tti, c)

  def typeInsn(op: Int, tti: TypeInfo[_]): ValueX = new TypeInsnX(op, tti)

  def typeInsn(op: Int, tti: TypeInfo[_], v: ValueX): ValueX = {
    val x = new TypeInsnX(op, tti)
    setChildren(x, v)
    x
  }

  def methodStmt(
    op: Int, owner: String, name: String, desc: String, isInterface: Boolean,
    returnTypeInfo: TypeInfo[_],
    args: IndexedSeq[ValueX]
  ): StmtX = {
    val x = new MethodStmtX(op, new MethodLit(owner, name, desc, isInterface, returnTypeInfo))
    setChildren(x, args)
    x
  }

  def methodStmt(
    op: Int, method: Method, args: IndexedSeq[ValueX]
  ): StmtX = {
    val x = new MethodStmtX(op, method)
    setChildren(x, args)
    x
  }

  def methodInsn(
    op: Int, owner: String, name: String, desc: String, isInterface: Boolean,
    returnTypeInfo: TypeInfo[_],
    args: IndexedSeq[ValueX]
  ): ValueX = {
    val x = new MethodX(op, new MethodLit(owner, name, desc, isInterface, returnTypeInfo))
    setChildren(x, args)
    x
  }

  def methodInsn(
    op: Int, m: MethodRef, args: IndexedSeq[ValueX]
  ): ValueX = {
    val x = new MethodX(op, m)
    setChildren(x, args)
    x
  }

  def getStaticField(owner: String, name: String, ti: TypeInfo[_]): ValueX =
    new GetFieldX(GETSTATIC, new FieldLit(owner, name, ti))

  def getStaticField(lf: StaticField): ValueX =
    new GetFieldX(GETSTATIC, lf)

  def getField(owner: String, name: String, ti: TypeInfo[_], obj: ValueX): ValueX = {
    val x = new GetFieldX(GETFIELD, new FieldLit(owner, name, ti))
    setChildren(x, obj)
    x
  }

  def getField(owner: String, name: String, ti: TypeInfo[_]): (ValueX) => ValueX =
    (obj) => getField(owner, name, ti, obj)

  def getField(lf: Field): (ValueX) => ValueX = (obj) => getField(lf, obj)

  def getField(lf: Field, obj: ValueX): ValueX = {
    val x = new GetFieldX(GETFIELD, lf)
    setChildren(x, obj)
    x
  }

  def putStaticField(owner: String, name: String, ti: TypeInfo[_]): (ValueX) => StmtX =
    (c) => putStaticField(owner, name, ti, c)

  def putStaticField(owner: String, name: String, ti: TypeInfo[_], v: ValueX): StmtX = {
    val x = new PutFieldX(PUTSTATIC, new FieldLit(owner, name, ti))
    setChildren(x, v)
    x
  }

  def putStaticField(lf: StaticField, v: ValueX): StmtX = {
    val x = new PutFieldX(PUTSTATIC, lf)
    setChildren(x, v)
    x
  }

  def putField(owner: String, name: String, ti: TypeInfo[_], obj: ValueX, v: ValueX): StmtX = {
    val x = new PutFieldX(PUTFIELD, new FieldLit(owner, name, ti))
    setChildren(x, obj, v)
    x
  }

  def putField(owner: String, name: String, ti: TypeInfo[_]): (ValueX, ValueX) => StmtX =
    (obj, v) => putField(owner, name, ti, obj, v)

  def putField(f: Field, obj: ValueX, v: ValueX): StmtX = {
    val x = new PutFieldX(PUTFIELD, f)
    setChildren(x, obj, v)
    x
  }

  def ldcInsn(a: Any, ti: TypeInfo[_]): ValueX = new LdcX(a, ti)

  def returnx(): ControlX = new ReturnX()

  def returnx1(): (ValueX) => ControlX = (c) => returnx(c)

  def returnx(c: ValueX): ControlX = {
    val x = new ReturnX()
    setChildren(x, c)
    x
  }

  def newInstance(
    ti: TypeInfo[_],
    owner: String, name: String, desc: String, returnTypeInfo: TypeInfo[_],
    args: IndexedSeq[ValueX]
  ): ValueX =
    newInstance(ti, owner, name, desc, returnTypeInfo, args, 0)

  def newInstance(
    ti: TypeInfo[_],
    owner: String, name: String, desc: String, returnTypeInfo: TypeInfo[_],
    args: IndexedSeq[ValueX],
    lineNumber: Int
  ): ValueX = {
    val x = new NewInstanceX(ti, new MethodLit(owner, name, desc, isInterface = false, returnTypeInfo), lineNumber)
    setChildren(x, args)
    x
  }

  def newInstance(ti: TypeInfo[_], method: Method, args: IndexedSeq[ValueX]): ValueX =
    newInstance(ti, method, args, 0)

  def newInstance(ti: TypeInfo[_], method: Method, args: IndexedSeq[ValueX], lineNumber: Int): ValueX = {
    val x = new NewInstanceX(ti, method, lineNumber)
    setChildren(x, args)
    x
  }

  def checkcast(tti: TypeInfo[_]): (ValueX) => ValueX = (c) => checkcast(tti, c)

  def checkcast(tti: TypeInfo[_], c: ValueX): ValueX = typeInsn(CHECKCAST, tti, c: ValueX)

  def newArray(tti: TypeInfo[_]): (ValueX) => ValueX = (len) => newArray(len, tti)

  def newArray(len: ValueX, eti: TypeInfo[_]): ValueX = {
    val x = new NewArrayX(eti)
    setChildren(x, len)
    x
  }

  def defaultValue(ti: TypeInfo[_]): ValueX = ti match {
    case BooleanInfo => ldcInsn(0, ti)
    case IntInfo => ldcInsn(0, ti)
    case LongInfo => ldcInsn(0L, ti)
    case FloatInfo => ldcInsn(0.0f, ti)
    case DoubleInfo => ldcInsn(0.0, ti)
    case _: ClassInfo[_] => insn0(ACONST_NULL, ti)
    case _: ArrayInfo[_] => insn0(ACONST_NULL, ti)
  }
}
