package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual._

import scala.language.{dynamics, implicitConversions}

object IRBuilder {
  type E = Env[Type]

  implicit def funcToIRProxy(ir: E => IR): IRProxy = new IRProxy(ir)

  implicit def tableIRToProxy(tir: TableIR): TableIRProxy =
    new TableIRProxy(tir)

  implicit def irToProxy(ir: IR): IRProxy = (_: E) => ir

  implicit def intToProxy(i: Int): IRProxy = I32(i)

  implicit def booleanToProxy(b: Boolean): IRProxy = if (b) True() else False()

  def ref(s: Sym): IRProxy = (env: E) =>
    Ref(s, env.lookup(s))

  implicit def symToSymProxy(s: Sym): SymProxy = new SymProxy(s)

  implicit def arrayToProxy(seq: Seq[IRProxy]): IRProxy = (env: E) => {
    val irs = seq.map(_(env))
    val elType = irs.head.typ
    MakeArray(irs, TArray(elType))
  }

  implicit def arrayIRToProxy(seq: Seq[IR]): IRProxy = arrayToProxy(seq.map(irToProxy))

  def irRange(start: IRProxy, end: IRProxy, step: IRProxy = 1): IRProxy = (env: E) =>
    ArrayRange(start(env), end(env), step(env))

  def irArrayLen(a: IRProxy): IRProxy = (env: E) => ArrayLen(a(env))


  def irIf(cond: IRProxy)(cnsq: IRProxy)(altr: IRProxy): IRProxy = (env: E) =>
    If(cond(env), cnsq(env), altr(env))

  def makeArray(first: IRProxy, rest: IRProxy*): IRProxy = arrayToProxy(first +: rest)

  def makeStruct(fields: (Sym, IRProxy)*): IRProxy = (env: E) =>
    MakeStruct(fields.map { case (s, ir) => (s, ir(env)) })

  def makeTuple(values: IRProxy*): IRProxy = (env: E) =>
    MakeTuple(values.map(_(env)))

  class TableIRProxy(val tir: TableIR) extends AnyVal {
    def empty: E = Env.empty
    def globalEnv: E = typ.globalEnv
    def env: E = typ.rowEnv

    def typ: TableType = tir.typ

    def getGlobals: IR = TableGetGlobals(tir)

    def mapGlobals(newGlobals: IRProxy): TableIR =
      TableMapGlobals(tir, newGlobals(globalEnv))

    def mapRows(newRow: IRProxy): TableIR =
      TableMapRows(tir, newRow(env))

    def keyBy(keys: IndexedSeq[Sym], isSorted: Boolean = false): TableIR =
      TableKeyBy(tir, keys, isSorted)

    def rename(rowMap: Map[Sym, Sym], globalMap: Map[Sym, Sym] = Map.empty): TableIR =
      TableRename(tir, rowMap, globalMap)

    def renameGlobals(globalMap: Map[Sym, Sym]): TableIR =
      rename(Map.empty, globalMap)

    def filter(ir: IRProxy): TableIR =
      TableFilter(tir, ir(env))
  }

  class IRProxy(val ir: E => IR) extends Dynamic {
    def apply(idx: Any): IRProxy = (env: E) => idx match {
      case idx: IRProxy =>
        ArrayRef(ir(env), idx(env))
      case lookup: Sym =>
        val eval = ir(env)
        eval.typ match {
          case _: TStruct =>
            GetField(eval, lookup)
          case _: TArray =>
            ArrayRef(ir(env), ref(lookup)(env))
        }
    }

    def selectDynamic(field: String): IRProxy = (env: E) =>
      GetField(ir(env), Identifier(field))

    def unary_! = new IRProxy((env: E) => ApplyUnaryPrimOp(Bang(), ir(env)))

    def typecheck(t: Type): IRProxy = (env: E) => {
      val eval = ir(env)
      TypeCheck(eval, env, None)
      assert(eval.typ == t, t._toPretty + " " + eval.typ._toPretty)
      eval
    }

    def insertFields(fields: (Sym, IRProxy)*): IRProxy = (env: E) =>
      InsertFields(ir(env), fields.map { case (s, fir) => (s, fir(env)) })

    def selectFields(fields: Sym*): IRProxy = (env: E) =>
      SelectFields(ir(env), fields)

    def dropFields(fields: Sym*): IRProxy = (env: E) => {
      val struct = ir(env)
      val typ = struct.typ.asInstanceOf[TStruct]
      SelectFields(struct, typ.fieldNames.diff(fields))
    }

    def len: IRProxy = (env: E) => ArrayLen(ir(env))

    def filter(pred: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ArrayFilter(array, pred.s, pred.body(env.bind(pred.s -> eltType)))
    }

    def map(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ArrayMap(array, f.s, f.body(env.bind(f.s-> eltType)))
    }

    def flatMap(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ArrayFlatMap(array, f.s, f.body(env.bind(f.s-> eltType)))
    }

    def sort(ascending: IRProxy, onKey: Boolean = false): IRProxy = (env: E) => ArraySort(ir(env), ascending(env), onKey)

    def groupByKey: IRProxy = (env: E) => GroupByKey(ir(env))

    def toArray: IRProxy = (env: E) => ToArray(ir(env))

    def parallelize(nPartitions: Option[Int] = None): TableIR = TableParallelize(ir(Env.empty), nPartitions)

    private[ir] def apply(env: E): IR = ir(env)
  }

  class LambdaProxy(val s: Sym, val body: IRProxy)

  class SymProxy(val s: Sym) extends IRProxy(ref(s).ir) {
    def ~> (body: IRProxy): LambdaProxy = new LambdaProxy(s, body)
  }

  case class BindingProxy(s: Sym, value: IRProxy)

  object LetProxy {
    def bind(bindings: Seq[BindingProxy], body: IRProxy, env: E): IR =
      bindings match {
        case BindingProxy(sym, binding) +: rest =>
          val name = sym
          val value = binding(env)
          Let(name, value, bind(rest, body, env.bind(name -> value.typ)))
        case Seq() =>
          body(env)
      }
  }

  object let {
    def apply(args: LambdaProxy*): LetProxy = {
      new LetProxy(args.map(arg => BindingProxy(arg.s, arg.body)))
    }
  }

  class LetProxy(val bindings: Seq[BindingProxy]) extends AnyVal with Dynamic {
    def apply(body: IRProxy): IRProxy = in(body)

    def in(body: IRProxy): IRProxy = { (env: E) =>
      LetProxy.bind(bindings, body, env)
    }
  }
}
