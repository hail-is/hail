package is.hail

import is.hail.asm4s.{ArrayInfo, BooleanInfo, ClassInfo, DoubleInfo, FloatInfo, IntInfo, LongInfo, TypeInfo, UnitInfo}
import is.hail.utils.FastIndexedSeq
import org.objectweb.asm.Opcodes._

package object lir {
  var counter: Long = 0

  def genName(tag: String, baseName: String): String = {
    counter += 1
    if (baseName != null)
      s"__$tag$counter$baseName"
    else
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
    x.setDefault(Ldefault)
    x.setCases(cases)
    x
  }

  def goto(L: Block): ControlX = {
    assert(L != null)
    val x = new GotoX
    x.setArity(0)
    x.setL(L)
    x
  }

  def store(l: Local): (ValueX) => StmtX = (c) => store(l, c)

  def store(l: Local, c: ValueX): StmtX = {
    val x = new StoreX(l)
    setChildren(x, c)
    x
  }

  def iincInsn(l: Local, i: Int): StmtX = new IincX(l, i)

  def insn1(op: Int): (ValueX) => ValueX = (c) => insn(op, c)

  def insn1(op: Int, _ti: TypeInfo[_]): (ValueX) => ValueX = (c) => insn(op, _ti, c)

  def insn2(op: Int): (ValueX, ValueX) => ValueX = (c1, c2) => insn(op, c1, c2)

  def insn3(op: Int): (ValueX, ValueX, ValueX) => ValueX = (c1, c2, c3) => insn(op, c1, c2, c3)

  def insn(op: Int, _ti: TypeInfo[_], args: IndexedSeq[ValueX]): ValueX = {
    val x = new InsnX(op, _ti)
    setChildren(x, args)
    x
  }

  def insn(op: Int): ValueX = insn(op, null, FastIndexedSeq.empty)

  def insn(op: Int, _ti: TypeInfo[_]): ValueX = insn(op, _ti, FastIndexedSeq.empty)

  def insn(op: Int, c: ValueX): ValueX = insn(op, null, FastIndexedSeq(c))

  def insn(op: Int, _ti: TypeInfo[_], c: ValueX): ValueX = insn(op, _ti, FastIndexedSeq(c))

  def insn(op: Int, c1: ValueX, c2: ValueX): ValueX = insn(op, null, FastIndexedSeq(c1, c2))

  def insn(op: Int, c1: ValueX, c2: ValueX, c3: ValueX): ValueX = insn(op, null, FastIndexedSeq(c1, c2, c3))

  def stmtOp3(op: Int): (ValueX, ValueX, ValueX) => StmtX = (c1, c2, c3) => stmtOp(op, c1, c2, c3)

  def stmtOp(op: Int, args: IndexedSeq[ValueX]): StmtX = {
    val x = new StmtOpX(op)
    setChildren(x, args)
    x
  }

  def throwx(c: ValueX): ControlX = {
    val x = new ThrowX()
    setChildren(x, c)
    x
  }

  def stmtOp(op: Int, c1: ValueX, c2: ValueX, c3: ValueX): StmtX = stmtOp(op, FastIndexedSeq(c1, c2, c3))

  def load(l: Local): ValueX = new LoadX(l)

  def typeInsn1(op: Int, t: String): (ValueX) => ValueX = (c) => typeInsn(op, t, c)

  def typeInsn(op: Int, t: String): ValueX = new TypeInsnX(op, t)

  def typeInsn(op: Int, t: String, v: ValueX): ValueX = {
    val x = new TypeInsnX(op, t)
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

  def ldcInsn(a: Any): ValueX = new LdcX(a)

  def returnx(): ControlX = new ReturnX()

  def returnx1(): (ValueX) => ControlX = (c) => returnx(c)

  def returnx(c: ValueX): ControlX = {
    val x = new ReturnX()
    setChildren(x, c)
    x
  }

  def newInstance(
    ti: TypeInfo[_]
  ): ValueX = new NewInstanceX(ti)

  def checkcast(iname: String): (ValueX) => ValueX = (c) => checkcast(iname, c)

  def checkcast(iname: String, c: ValueX): ValueX = typeInsn(CHECKCAST, iname, c: ValueX)

  def newArray(tti: TypeInfo[_]): (ValueX) => ValueX = (len) => newArray(len, tti)

  def newArray(len: ValueX, eti: TypeInfo[_]): ValueX = {
    val x = new NewArrayX(eti)
    setChildren(x, len)
    x
  }

  def defaultValue(ti: TypeInfo[_]): ValueX = ti match {
    case BooleanInfo => ldcInsn(false)
    case IntInfo => ldcInsn(0)
    case LongInfo => ldcInsn(0L)
    case FloatInfo => ldcInsn(0.0f)
    case DoubleInfo => ldcInsn(0.0)
    case _: ClassInfo[_] => insn(ACONST_NULL, ti)
    case _: ArrayInfo[_] => insn(ACONST_NULL, ti)
  }
}
