package is.hail.expr.ir

import is.hail.expr.types.{TArray, TStruct, TableType, Type}

import scala.language.{dynamics, implicitConversions}

object IRBuilder {
  type E = Env[Type]

  implicit def funcToIRProxy(ir: E => IR): IRProxy = new IRProxy(ir)

  implicit def tableIRToProxy(tir: TableIR): TableIRProxy =
    new TableIRProxy(tir)

  implicit def irToProxy(ir: IR): IRProxy = (_: E) => ir

  implicit def intToProxy(i: Int): IRProxy = I32(i)

  implicit def ref(s: Symbol): IRProxy = (env: E) =>
    Ref(s.name, env.lookup(s.name))

  implicit def symbolToSymbolProxy(s: Symbol): SymbolProxy = new SymbolProxy(s)

  implicit def arrayToProxy(seq: Seq[IRProxy]): IRProxy = (env: E) => {
    val irs = seq.map(_(env))
    val elType = irs.head.typ
    MakeArray(irs, TArray(elType))
  }

  implicit def arrayIRToProxy(seq: Seq[IR]): IRProxy = arrayToProxy(seq.map(irToProxy))

  def irRange(start: IRProxy, end: IRProxy, step: IRProxy = 1): IRProxy = (env: E) =>
    ArrayRange(start(env), end(env), step(env))

  def irIf(cond: IRProxy)(cnsq: IRProxy)(altr: IRProxy): IRProxy = (env: E) =>
    If(cond(env), cnsq(env), altr(env))

  def makeStruct(fields: (Symbol, IRProxy)*): IRProxy = (env: E) =>
    MakeStruct(fields.map { case (s, ir) => (s.name, ir(env)) })

  def makeTuple(values: IRProxy*): IRProxy = (env: E) =>
    MakeTuple(values.map(_(env)))

  class TableIRProxy(val tir: TableIR) extends AnyVal {
    def empty: E = Env.empty
    def globalEnv: E = typ.globalEnv
    def env: E = typ.rowEnv

    def typ: TableType = tir.typ

    def mapGlobals(newGlobals: IRProxy): TableIR =
      TableMapGlobals(tir, newGlobals(globalEnv))

    def mapRows(newRow: IRProxy): TableIR =
      TableMapRows(tir, newRow(env))

    def keyBy(keys: IndexedSeq[String], isSorted: Boolean = false): TableIR =
      TableKeyBy(tir, keys, isSorted)

    def rename(rowMap: Map[String, String], globalMap: Map[String, String] = Map.empty): TableIR =
      TableRename(tir, rowMap, globalMap)

    def renameGlobals(globalMap: Map[String, String]): TableIR =
      rename(Map.empty, globalMap)

    def filter(ir: IRProxy): TableIR =
      TableFilter(tir, ir(env))
  }

  class IRProxy(val ir: E => IR) extends AnyVal with Dynamic {
    def <=: (s: Symbol): BindingProxy = BindingProxy(s, this)

    def apply(idx: IRProxy): IRProxy = (env: E) =>
      ArrayRef(ir(env), idx(env))

    def selectDynamic(field: String): IRProxy = (env: E) =>
      GetField(ir(env), field)

    def apply(lookup: Symbol): IRProxy = (env: E) => {
      val eval = ir(env)
      eval.typ match {
        case _: TStruct =>
          GetField(eval, lookup.name)
        case _: TArray =>
          ArrayRef(ir(env), ref(lookup)(env))
      }
    }

    def typecheck(t: Type): IRProxy = (env: E) => {
      val eval = ir(env)
      TypeCheck(eval, env, None)
      assert(eval.typ == t, t._toPretty + " " + eval.typ._toPretty)
      eval
    }

    def insertFields(fields: (Symbol, IRProxy)*): IRProxy = (env: E) =>
      InsertFields(ir(env), fields.map { case (s, fir) => (s.name, fir(env)) })

    def selectFields(fields: String*): IRProxy = (env: E) =>
      SelectFields(ir(env), fields)

    def dropFields(fields: Symbol*): IRProxy = (env: E) => {
      val struct = ir(env)
      val typ = struct.typ.asInstanceOf[TStruct]
      SelectFields(struct, typ.fieldNames.diff(fields.map(_.name)))
    }

    def len: IRProxy = (env: E) => ArrayLen(ir(env))

    def filter(pred: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ArrayFilter(array, pred.s.name, pred.body(env.bind(pred.s.name -> eltType)))
    }

    def map(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ArrayMap(ir(env), f.s.name, f.body(env.bind(f.s.name -> eltType)))
    }

    def groupByKey: IRProxy = (env: E) => GroupByKey(ir(env))

    def toArray: IRProxy = (env: E) => ToArray(ir(env))

    private[ir] def apply(env: E): IR = ir(env)
  }

  class LambdaProxy(val s: Symbol, val body: IRProxy)

  class SymbolProxy(val s: Symbol) extends AnyVal {
    def ~> (body: IRProxy): LambdaProxy = new LambdaProxy(s, body)
  }

  case class BindingProxy(s: Symbol, value: IRProxy)

  object LetProxy {
    def bind(bindings: Seq[BindingProxy], body: IRProxy, env: E): IR =
      bindings match {
        case BindingProxy(sym, binding) +: rest =>
          val name = sym.name
          val value = binding(env)
          Let(name, value, bind(rest, body, env.bind(name -> value.typ)))
        case Seq() =>
          body(env)
      }
  }

  object let extends Dynamic {
    def applyDynamicNamed(method: String)(args: (String, Any)*): LetProxy = {
      assert(method == "apply")
      new LetProxy(args.map { case (s, b) => BindingProxy(Symbol(s), b.asInstanceOf[IRProxy]) })
    }
  }

  class LetProxy(val bindings: Seq[BindingProxy]) extends AnyVal with Dynamic {
    def apply(body: IRProxy): IRProxy = in(body)

    def in(body: IRProxy): IRProxy = { (env: E) =>
      LetProxy.bind(bindings, body, env)
    }
  }
}
