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

  def ifx(op: Int, c: ValueX, Ltrue: Block, Lfalse: Block, lineNumber: Int): ControlX = {
    val x = new IfX(op, lineNumber)
    setChildren(x, c)
    x.setLtrue(Ltrue)
    x.setLfalse(Lfalse)
    x
  }

  def ifx(op: Int, c1: ValueX, c2: ValueX, Ltrue: Block, Lfalse: Block, lineNumber: Int): ControlX = {
    val x = new IfX(op, lineNumber)
    setChildren(x, c1, c2)
    x.setLtrue(Ltrue)
    x.setLfalse(Lfalse)
    x
  }

  def switch(c: ValueX, Ldefault: Block, cases: IndexedSeq[Block], lineNumber: Int): ControlX = {
    if (cases.isEmpty)
      return goto(Ldefault, lineNumber)

    val x = new SwitchX(lineNumber)
    setChildren(x, c)
    x.setLdefault(Ldefault)
    x.setLcases(cases)
    x
  }

  def goto(L: Block, lineNumber: Int): ControlX = {
    assert(L != null)
    val x = new GotoX(lineNumber)
    x.setArity(0)
    x.setL(L)
    x
  }

  def store(l: Local, c: ValueX, lineNumber: Int): StmtX = {
    val x = new StoreX(l, lineNumber)
    setChildren(x, c)
    x
  }

  def iincInsn(l: Local, i: Int, lineNumber: Int ): StmtX = new IincX(l, i, lineNumber)

  def insn0(op: Int, lineNumber: Int): ValueX = insn(op, null, FastIndexedSeq.empty, lineNumber)

  def insn0(op: Int, _ti: TypeInfo[_], lineNumber: Int): ValueX = insn(op, _ti, FastIndexedSeq.empty, lineNumber)

  def insn1(op: Int, lineNumber: Int): (ValueX) => ValueX = (c) =>
    insn(op, null, FastIndexedSeq(c), lineNumber)

  def insn1(op: Int, _ti: TypeInfo[_], lineNumber: Int): (ValueX) => ValueX = (c) =>
    insn(op, _ti, FastIndexedSeq(c), lineNumber)

  def insn2(op: Int, lineNumber: Int): (ValueX, ValueX) => ValueX = (c1, c2) =>
    insn(op, null, FastIndexedSeq(c1, c2), lineNumber)

  def insn3(op: Int, lineNumber: Int): (ValueX, ValueX, ValueX) => ValueX = (c1, c2, c3) =>
    insn(op, null, FastIndexedSeq(c1, c2, c3), lineNumber)

  def insn(op: Int, _ti: TypeInfo[_], args: IndexedSeq[ValueX], lineNumber: Int): ValueX = {
    val x = new InsnX(op, _ti, lineNumber)
    setChildren(x, args)
    x
  }

  def stmtOp3(op: Int, lineNumber: Int): (ValueX, ValueX, ValueX) => StmtX = (c1, c2, c3) => stmtOp(op, c1, c2, c3, lineNumber)

  def stmtOp(op: Int, args: IndexedSeq[ValueX], lineNumber: Int): StmtX = {
    val x = new StmtOpX(op, lineNumber)
    setChildren(x, args)
    x
  }

  def throwx(c: ValueX, lineNumber: Int = 0): ControlX = {
    val x = new ThrowX(lineNumber)
    setChildren(x, c)
    x
  }

  def stmtOp(op: Int, c1: ValueX, c2: ValueX, c3: ValueX, lineNumber: Int): StmtX =
    stmtOp(op, FastIndexedSeq(c1, c2, c3), lineNumber)

  def load(l: Local, lineNumber: Int): ValueX = new LoadX(l, lineNumber)

  def typeInsn1(op: Int, t: String, lineNumber: Int): (ValueX) => ValueX = (c) => typeInsn(op, t, c, lineNumber)

  def typeInsn(op: Int, t: String, lineNumber: Int): ValueX = new TypeInsnX(op, t, lineNumber)

  def typeInsn(op: Int, t: String, v: ValueX, lineNumber: Int): ValueX = {
    val x = new TypeInsnX(op, t, lineNumber)
    setChildren(x, v)
    x
  }

  def methodStmt(
    op: Int, owner: String, name: String, desc: String, isInterface: Boolean,
    returnTypeInfo: TypeInfo[_],
    args: IndexedSeq[ValueX],
    lineNumber: Int
  ): StmtX = {
    val x = new MethodStmtX(op, new MethodLit(owner, name, desc, isInterface, returnTypeInfo), lineNumber)
    setChildren(x, args)
    x
  }

  def methodStmt(
    op: Int, method: Method, args: IndexedSeq[ValueX], lineNumber: Int
  ): StmtX = {
    val x = new MethodStmtX(op, method, lineNumber)
    setChildren(x, args)
    x
  }

  def methodInsn(
    op: Int, owner: String, name: String, desc: String, isInterface: Boolean,
    returnTypeInfo: TypeInfo[_],
    args: IndexedSeq[ValueX],
    lineNumber: Int
  ): ValueX = {
    val x = new MethodX(op, new MethodLit(owner, name, desc, isInterface, returnTypeInfo), lineNumber)
    setChildren(x, args)
    x
  }

  def methodInsn(
    op: Int, m: MethodRef, args: IndexedSeq[ValueX], lineNumber: Int
  ): ValueX = {
    val x = new MethodX(op, m, lineNumber)
    setChildren(x, args)
    x
  }

  def getStaticField(owner: String, name: String, ti: TypeInfo[_], lineNumber: Int): ValueX =
    new GetFieldX(GETSTATIC, new FieldLit(owner, name, ti), lineNumber)

  def getStaticField(lf: StaticField, lineNumber: Int): ValueX =
    new GetFieldX(GETSTATIC, lf, lineNumber)

  def getField(owner: String, name: String, ti: TypeInfo[_], obj: ValueX, lineNumber: Int): ValueX = {
    val x = new GetFieldX(GETFIELD, new FieldLit(owner, name, ti), lineNumber)
    setChildren(x, obj)
    x
  }

  def getField(owner: String, name: String, ti: TypeInfo[_], lineNumber: Int): (ValueX) => ValueX =
    (obj) => getField(owner, name, ti, obj, lineNumber)

  def getField(lf: Field, lineNumber: Int): (ValueX) => ValueX = (obj) => getField(lf, obj, lineNumber)

  def getField(lf: Field, obj: ValueX, lineNumber: Int): ValueX = {
    val x = new GetFieldX(GETFIELD, lf, lineNumber)
    setChildren(x, obj)
    x
  }

  def putStaticField(owner: String, name: String, ti: TypeInfo[_], lineNumber: Int): (ValueX) => StmtX =
    (c) => putStaticField(owner, name, ti, c, lineNumber)

  def putStaticField(owner: String, name: String, ti: TypeInfo[_], v: ValueX, lineNumber: Int): StmtX = {
    val x = new PutFieldX(PUTSTATIC, new FieldLit(owner, name, ti), lineNumber)
    setChildren(x, v)
    x
  }

  def putStaticField(lf: StaticField, v: ValueX, lineNumber: Int): StmtX = {
    val x = new PutFieldX(PUTSTATIC, lf, lineNumber)
    setChildren(x, v)
    x
  }

  def putField(owner: String, name: String, ti: TypeInfo[_], obj: ValueX, v: ValueX, lineNumber: Int): StmtX = {
    val x = new PutFieldX(PUTFIELD, new FieldLit(owner, name, ti), lineNumber)
    setChildren(x, obj, v)
    x
  }

  def putField(owner: String, name: String, ti: TypeInfo[_], lineNumber: Int): (ValueX, ValueX) => StmtX =
    (obj, v) => putField(owner, name, ti, obj, v, lineNumber)

  def putField(f: Field, obj: ValueX, v: ValueX, lineNumber: Int): StmtX = {
    val x = new PutFieldX(PUTFIELD, f, lineNumber)
    setChildren(x, obj, v)
    x
  }

  def ldcInsn(a: Any, ti: TypeInfo[_], lineNumber: Int): ValueX = new LdcX(a, ti, lineNumber)

  def returnx(lineNumber: Int): ControlX = new ReturnX(lineNumber)

  def returnx1(lineNumber: Int): (ValueX) => ControlX = (c) => returnx(c, lineNumber)

  def returnx(c: ValueX, lineNumber: Int): ControlX = {
    val x = new ReturnX(lineNumber)
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

  def checkcast(iname: String, lineNumber: Int): (ValueX) => ValueX = (c) => checkcast(iname, c, lineNumber)

  def checkcast(iname: String, c: ValueX, lineNumber: Int): ValueX = typeInsn(CHECKCAST, iname, c: ValueX, lineNumber)

  def newArray(tti: TypeInfo[_], lineNumber: Int): (ValueX) => ValueX = (len) => newArray(len, tti, lineNumber)

  def newArray(len: ValueX, eti: TypeInfo[_], lineNumber: Int): ValueX = {
    val x = new NewArrayX(eti, lineNumber)
    setChildren(x, len)
    x
  }

  def defaultValue(ti: TypeInfo[_], lineNumber: Int): ValueX = ti match {
    case BooleanInfo => ldcInsn(0, ti, lineNumber)
    case IntInfo => ldcInsn(0, ti, lineNumber)
    case LongInfo => ldcInsn(0L, ti, lineNumber)
    case FloatInfo => ldcInsn(0.0f, ti, lineNumber)
    case DoubleInfo => ldcInsn(0.0, ti, lineNumber)
    case _: ClassInfo[_] => insn0(ACONST_NULL, ti, lineNumber)
    case _: ArrayInfo[_] => insn0(ACONST_NULL, ti, lineNumber)
  }
}
