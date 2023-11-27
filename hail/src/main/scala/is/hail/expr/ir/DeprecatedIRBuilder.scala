package is.hail.expr.ir

import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

import scala.language.{dynamics, implicitConversions}

object DeprecatedIRBuilder {
  type E = Env[Type]

  implicit def funcToIRProxy(ir: E => IR): IRProxy = new IRProxy(ir)

  implicit def tableIRToProxy(tir: TableIR): TableIRProxy =
    new TableIRProxy(tir)

  implicit def irToProxy(ir: IR): IRProxy = (_: E) => ir

  implicit def strToProxy(s: String): IRProxy = Str(s)

  implicit def intToProxy(i: Int): IRProxy = I32(i)

  implicit def booleanToProxy(b: Boolean): IRProxy = if (b) True() else False()

  implicit def ref(s: Symbol): IRProxy = (env: E) =>
    Ref(s.name, env.lookup(s.name))

  implicit def symbolToSymbolProxy(s: Symbol): SymbolProxy = new SymbolProxy(s)

  implicit def arrayToProxy(seq: IndexedSeq[IRProxy]): IRProxy = (env: E) => {
    val irs = seq.map(_ (env))
    val elType = irs.head.typ
    MakeArray(irs, TArray(elType))
  }

  implicit def arrayIRToProxy(seq: IndexedSeq[IR]): IRProxy = arrayToProxy(seq.map(irToProxy))

  def irRange(start: IRProxy, end: IRProxy, step: IRProxy = 1): IRProxy = (env: E) =>
    ToArray(StreamRange(start(env), end(env), step(env)))

  def irArrayLen(a: IRProxy): IRProxy = (env: E) => ArrayLen(a(env))


  def irIf(cond: IRProxy)(cnsq: IRProxy)(altr: IRProxy): IRProxy = (env: E) =>
    If(cond(env), cnsq(env), altr(env))

  def irDie(message: IRProxy, typ: Type): IRProxy = (env: E) =>
    Die(message(env), typ, -1)

  def makeArray(first: IRProxy, rest: IRProxy*): IRProxy =
    arrayToProxy((first +: rest).toArray[IRProxy])

  def makeStruct(fields: (Symbol, IRProxy)*): IRProxy = (env: E) =>
    MakeStruct(fields.toArray.map { case (s, ir) => (s.name, ir(env)) })

  def concatStructs(struct1: IRProxy, struct2: IRProxy): IRProxy = (env: E) => {
    val s2Type = struct2(env).typ.asInstanceOf[TStruct]
    let(__struct2 = struct2) {
      struct1.insertFields(s2Type.fieldNames.map(f => Symbol(f) -> '__struct2(Symbol(f))): _*)
    }(env)
  }

  def makeTuple(values: IRProxy*): IRProxy = (env: E) =>
    MakeTuple.ordered(values.toArray.map(_ (env)))

  def applyAggOp(
                  op: AggOp,
                  initOpArgs: IndexedSeq[IRProxy] = FastSeq(),
                  seqOpArgs: IndexedSeq[IRProxy] = FastSeq()): IRProxy = (env: E) => {

    val i = initOpArgs.map(x => x(env))
    val s = seqOpArgs.map(x => x(env))

    ApplyAggOp(i, s, AggSignature(op, i.map(_.typ), s.map(_.typ)))
  }

  def aggFilter(filterCond: IRProxy, query: IRProxy, isScan: Boolean = false): IRProxy = (env: E) =>
    AggFilter(filterCond(env), query(env), isScan)

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

    def explode(sym: Symbol): TableIR = TableExplode(tir, FastSeq(sym.name))

    def aggregateByKey(aggIR: IRProxy): TableIR = TableAggregateByKey(tir, aggIR(env))

    def keyBy(keys: IndexedSeq[String], isSorted: Boolean = false): TableIR =
      TableKeyBy(tir, keys, isSorted)

    def rename(rowMap: Map[String, String], globalMap: Map[String, String] = Map.empty): TableIR =
      TableRename(tir, rowMap, globalMap)

    def renameGlobals(globalMap: Map[String, String]): TableIR =
      rename(Map.empty, globalMap)

    def filter(ir: IRProxy): TableIR =
      TableFilter(tir, ir(env))

    def distinct(): TableIR = TableDistinct(tir)

    def collect(): IRProxy = TableCollect(tir)

    def collectAsDict(): IRProxy = {
      val uid = genUID()
      val keyFields = tir.typ.key
      val valueFields = tir.typ.valueType.fieldNames
      keyBy(FastSeq())
        .collect()
        .apply('rows)
        .map(Symbol(uid) ~> makeTuple(Symbol(uid).selectFields(keyFields: _*), Symbol(uid).selectFields(valueFields: _*)))
        .toDict
    }

    def aggregate(ir: IRProxy): IR =
      TableAggregate(tir, ir(env))
  }

  class IRProxy(val ir: E => IR) extends AnyVal with Dynamic {
    def apply(idx: IRProxy): IRProxy = (env: E) =>
      ArrayRef(ir(env), idx(env))

    def invoke(name: String, rt: Type, args: IRProxy*): IRProxy = { env: E =>
      val irArgs = Array(ir(env)) ++ args.map(_ (env))
      is.hail.expr.ir.invoke(name, rt, irArgs: _*)
    }

    def selectDynamic(field: String): IRProxy = (env: E) =>
      GetField(ir(env), field)

    def +(other: IRProxy): IRProxy = (env: E) => ApplyBinaryPrimOp(Add(), ir(env), other(env))

    def -(other: IRProxy): IRProxy = (env: E) => ApplyBinaryPrimOp(Subtract(), ir(env), other(env))

    def *(other: IRProxy): IRProxy = (env: E) => ApplyBinaryPrimOp(Multiply(), ir(env), other(env))

    def /(other: IRProxy): IRProxy = (env: E) => ApplyBinaryPrimOp(FloatingPointDivide(), ir(env), other(env))

    def floorDiv(other: IRProxy): IRProxy = (env: E) => ApplyBinaryPrimOp(RoundToNegInfDivide(), ir(env), other(env))

    def &&(other: IRProxy): IRProxy = invoke("land", TBoolean, ir, other)

    def ||(other: IRProxy): IRProxy = invoke("lor", TBoolean, ir, other)

    def toI: IRProxy = (env: E) => Cast(ir(env), TInt32)

    def toL: IRProxy = (env: E) => Cast(ir(env), TInt64)

    def toF: IRProxy = (env: E) => Cast(ir(env), TFloat32)

    def toD: IRProxy = (env: E) => Cast(ir(env), TFloat64)

    def unary_-(): IRProxy = (env: E) => ApplyUnaryPrimOp(Negate, ir(env))

    def unary_!(): IRProxy = (env: E) => ApplyUnaryPrimOp(Bang, ir(env))

    def ceq(other: IRProxy): IRProxy = (env: E) => {
      val left = ir(env)
      val right = other(env);
      ApplyComparisonOp(EQWithNA(left.typ, right.typ), left, right)
    }

    def cne(other: IRProxy): IRProxy = (env: E) => {
      val left = ir(env)
      val right = other(env)
      ApplyComparisonOp(NEQWithNA(left.typ, right.typ), left, right)
    }

    def <(other: IRProxy): IRProxy = (env: E) => {
      val left = ir(env)
      val right = other(env)
      ApplyComparisonOp(LT(left.typ, right.typ), left, right)
    }

    def >(other: IRProxy): IRProxy = (env: E) => {
      val left = ir(env)
      val right = other(env)
      ApplyComparisonOp(GT(left.typ, right.typ), left, right)
    }

    def <=(other: IRProxy): IRProxy = (env: E) => {
      val left = ir(env)
      val right = other(env)
      ApplyComparisonOp(LTEQ(left.typ, right.typ), left, right)
    }

    def >=(other: IRProxy): IRProxy = (env: E) => {
      val left = ir(env)
      val right = other(env)
      ApplyComparisonOp(GTEQ(left.typ, right.typ), left, right)
    }

    def apply(lookup: Symbol): IRProxy = (env: E) => {
      val eval = ir(env)
      eval.typ match {
        case _: TStruct =>
          GetField(eval, lookup.name)
        case _: TArray =>
          ArrayRef(ir(env), ref(lookup)(env))
      }
    }

    def castRename(t: Type): IRProxy = (env: E) => CastRename(ir(env), t)

    def insertFields(fields: (Symbol, IRProxy)*): IRProxy = insertFieldsList(fields)

    def insertFieldsList(fields: Seq[(Symbol, IRProxy)], ordering: Option[IndexedSeq[String]] = None): IRProxy = (env: E) =>
      InsertFields(ir(env), fields.map { case (s, fir) => (s.name, fir(env))}, ordering)

    def selectFields(fields: String*): IRProxy = (env: E) =>
      SelectFields(ir(env), fields.toArray[String])

    def dropFieldList(fields: IndexedSeq[String]): IRProxy = (env: E) => {
      val struct = ir(env)
      val typ = struct.typ.asInstanceOf[TStruct]
      SelectFields(struct, typ.fieldNames.diff(fields))
    }

    def dropFields(fields: Symbol*): IRProxy = dropFieldList(fields.map(_.name).toArray[String])

    def insertStruct(other: IRProxy, ordering: Option[IndexedSeq[String]] = None): IRProxy = (env: E) => {
      val right = other(env)
      val sym = genUID()
      Let(FastSeq(sym -> right),
        InsertFields(
          ir(env),
          right.typ.asInstanceOf[TStruct].fieldNames.map(f => f -> GetField(Ref(sym, right.typ), f)),
          ordering
        )
      )
    }

    def len: IRProxy = (env: E) => ArrayLen(ir(env))

    def isNA: IRProxy = (env: E) => IsNA(ir(env))

    def orElse(alt: IRProxy): IRProxy = { env: E =>
      val uid = genUID()
      val eir = ir(env)
      Let(FastSeq(uid -> eir), If(IsNA(Ref(uid, eir.typ)), alt(env), Ref(uid, eir.typ)))
    }

    def filter(pred: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ToArray(StreamFilter(ToStream(array), pred.s.name, pred.body(env.bind(pred.s.name -> eltType))))
    }

    def map(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ToArray(StreamMap(ToStream(array), f.s.name, f.body(env.bind(f.s.name -> eltType))))
    }

    def aggExplode(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      AggExplode(ToStream(array), f.s.name, f.body(env.bind(f.s.name, array.typ.asInstanceOf[TArray].elementType)), isScan = false)
    }

    def flatMap(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ToArray(StreamFlatMap(ToStream(array), f.s.name, ToStream(f.body(env.bind(f.s.name -> eltType)))))
    }

    def streamAgg(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      StreamAgg(ToStream(array), f.s.name, f.body(env.bind(f.s.name -> eltType)))
    }

    def streamAggScan(f: LambdaProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      ToArray(StreamAggScan(ToStream(array), f.s.name, f.body(env.bind(f.s.name -> eltType))))
    }

    def arraySlice(start: IRProxy, stop: Option[IRProxy], step: IRProxy): IRProxy = {
      (env: E) => ArraySlice(this.ir(env), start.ir(env), stop.map(inner => inner.ir(env)), step.ir(env), ErrorIDs.NO_ERROR)
    }

    def aggElements(elementsSym: Symbol, indexSym: Symbol, knownLength: Option[IRProxy])(aggBody: IRProxy): IRProxy = (env: E) => {
      val array = ir(env)
      val eltType = array.typ.asInstanceOf[TArray].elementType
      AggArrayPerElement(
        array,
        elementsSym.name,
        indexSym.name,
        aggBody.apply(env.bind(elementsSym.name -> eltType, indexSym.name -> TInt32)),
        knownLength.map(_ (env)),
        isScan = false)
    }

    def sort(ascending: IRProxy, onKey: Boolean = false): IRProxy = (env: E) => ArraySort(ToStream(ir(env)), ascending(env), onKey)

    def groupByKey: IRProxy = (env: E) => GroupByKey(ToStream(ir(env)))

    def toArray: IRProxy = (env: E) => ToArray(ToStream(ir(env)))

    def toDict: IRProxy = (env: E) => ToDict(ToStream(ir(env)))

    def parallelize(nPartitions: Option[Int] = None): TableIR = TableParallelize(ir(Env.empty), nPartitions)

    def arrayStructToDict(keyFields: IndexedSeq[String]): IRProxy = {
      val element = Symbol(genUID())
      ir
        .map(element ~>
          makeTuple(
            element.selectFields(keyFields: _*),
            element.dropFieldList(keyFields)))
        .toDict
    }

    def tupleElement(i: Int): IRProxy = (env: E) => GetTupleElement(ir(env), i)

    private[ir] def apply(env: E): IR = ir(env)
  }

  class LambdaProxy(val s: Symbol, val body: IRProxy)

  class SymbolProxy(val s: Symbol) extends AnyVal {
    def ~>(body: IRProxy): LambdaProxy = new LambdaProxy(s, body)
  }

  case class BindingProxy(s: Symbol, value: IRProxy)

  object LetProxy {
    def bind(bindings: Seq[BindingProxy], body: IRProxy, env: E, scope: Int): IR =
      bindings match {
        case BindingProxy(sym, binding) +: rest =>
          val name = sym.name
          val value = binding(env)
          scope match {
            case Scope.EVAL => Let(FastSeq(name -> value), bind(rest, body, env.bind(name -> value.typ), scope))
            case Scope.AGG => AggLet(name, value, bind(rest, body, env.bind(name -> value.typ), scope), isScan = false)
            case Scope.SCAN => AggLet(name, value, bind(rest, body, env.bind(name -> value.typ), scope), isScan = true)
          }
        case Seq() =>
          body(env)
      }
  }

  object let extends Dynamic {
    def applyDynamicNamed(method: String)(args: (String, IRProxy)*): LetProxy = {
      assert(method == "apply")
      new LetProxy(args.map { case (s, b) => BindingProxy(Symbol(s), b) })
    }
  }

  class LetProxy(val bindings: Seq[BindingProxy]) extends AnyVal with Dynamic {
    def apply(body: IRProxy): IRProxy = in(body)

    def in(body: IRProxy): IRProxy = { (env: E) =>
      LetProxy.bind(bindings, body, env, scope = Scope.EVAL)
    }
  }

  object aggLet extends Dynamic {
    def applyDynamicNamed(method: String)(args: (String, IRProxy)*): AggLetProxy = {
      assert(method == "apply")
      new AggLetProxy(args.map { case (s, b) => BindingProxy(Symbol(s), b) })
    }
  }

  class AggLetProxy(val bindings: Seq[BindingProxy]) extends AnyVal with Dynamic {
    def apply(body: IRProxy): IRProxy = in(body)

    def in(body: IRProxy): IRProxy = { (env: E) =>
      LetProxy.bind(bindings, body, env, scope = Scope.AGG)
    }
  }

  object MapIRProxy {
    def apply(f: (IRProxy) => IRProxy)(x: IRProxy): IRProxy = (e: E) => {
      MapIR(x => f(x)(e))(x(e))
    }
  }

  def subst(x: IRProxy, env: BindingEnv[IRProxy]): IRProxy = (e: E) => {
    Subst(x(e), BindingEnv(env.eval.mapValues(_ (e)),
      agg = env.agg.map(_.mapValues(_ (e))),
      scan = env.scan.map(_.mapValues(_ (e)))))
  }

  def lift(f: (IR) => IRProxy)(x: IRProxy): IRProxy = (e: E) => f(x(e))(e)
}
