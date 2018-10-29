package is.hail.expr.ir

import is.hail.expr.types.{TArray, TStruct, Type}

import scala.language.implicitConversions

object IRBuilder {
  type E = Env[Type]

  implicit def tableIRToProxy(tir: TableIR): TableIRProxy =
    new TableIRProxy(tir)

  implicit def funcToIRProxy(ir: E => IR): IRProxy = new IRProxy(ir)

  implicit def irToProxy(ir: IR): IRProxy = new IRProxy(_ => ir)

  implicit def intToProxy(i: Int): IRProxy = irToProxy(I32(i))

  implicit def ref(s: Symbol): IRProxy = new IRProxy( env =>
    Ref(s.name, env.lookup(s.name)))

  implicit def symbolToSymbolProxy(s: Symbol): SymbolProxy = new SymbolProxy(s)

  implicit def arrayToProxy(seq: Seq[IRProxy]): IRProxy = { (env: E) =>
    val irs = seq.map(_(env))
    val elType = irs.head.typ
    MakeArray(irs, TArray(elType))
  }

  implicit def arrayIRToProxy(seq: Seq[IR]): IRProxy = arrayToProxy(seq.map(irToProxy))

  def irRange(start: IRProxy, end: IRProxy, step: IRProxy = 1): IRProxy = (env: E) =>
    ArrayRange(start(env), end(env), step(env))

  def let(bindings: BindingProxy*): LetProxy = new LetProxy(bindings.toList)

  def irIf(cond: IRProxy)(cnsq: IRProxy)(altr: IRProxy): IRProxy = (env: E) =>
    If(cond(env), cnsq(env), altr(env))

  class TableIRProxy(val tir: TableIR) {
    def empty: E = Env.empty
    def globalEnv: E = Env("global" -> typ.globalType)
    def env: E = Env("row" -> typ.rowType, "global" -> typ.globalType)

    def typ = tir.typ

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

  class IRProxy(val ir: E => IR) {
    def <=: (s: Symbol): BindingProxy = new BindingProxy(s, ir)

    def apply(idx: IRProxy): IRProxy = (env: E) =>
      ArrayRef(ir(env), idx(env))

    def apply(lookup: Symbol): IRProxy = { (env: E) =>
      val eval = ir(env)
      eval.typ match {
        case _: TStruct =>
          GetField(eval, lookup.name)
        case _: TArray =>
          ArrayRef(ir(env), ref(lookup)(env))
      }
    }

    def insertFields(fields: (Symbol, IRProxy)*): IRProxy = (env: E) =>
      InsertFields(ir(env), fields.map { case (s, fir) => (s.name, fir(env)) })

    def selectFields(fields: String*): IRProxy = (env: E) =>
      SelectFields(ir(env), fields)

    def dropFields(fields: Symbol*): IRProxy = { (env: E) =>
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

    private[ir] def apply(env: E) = ir(env)
  }

  class LambdaProxy(val s: Symbol, val body: IRProxy)

  class SymbolProxy(val s: Symbol) extends AnyVal {
    def ~> (body: IRProxy): LambdaProxy = new LambdaProxy(s, body)
  }

  case class BindingProxy(s: Symbol, value: IRProxy)

  object LetProxy {
    def bind(bindings: List[BindingProxy], body: IRProxy, env: E): IR =
      bindings match {
        case BindingProxy(sym, binding) :: rest =>
          val name = sym.name
          val value = binding(env)
          Let(name, value, bind(rest, body, env.bind(name -> value.typ)))
        case Nil =>
          body(env)
      }
  }

  class LetProxy(bindings: List[BindingProxy]) {
    def in(body: IRProxy): IRProxy = { (env: E) =>
      LetProxy.bind(bindings, body, env)
    }
  }
}
