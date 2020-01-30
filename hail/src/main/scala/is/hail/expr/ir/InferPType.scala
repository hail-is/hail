package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TNDArray, TVoid}
import is.hail.utils._

object InferPType {
  def getNestedElementPTypes(ptypes: Seq[PType]): PType = {
    assert(ptypes.forall(_.virtualType.isOfType(ptypes.head.virtualType)))
    getNestedElementPTypesOfSameType(ptypes: Seq[PType])
  }

  def getNestedElementPTypesOfSameType(ptypes: Seq[PType]): PType = {
    ptypes.head match {
      case x: PStreamable => {
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PStreamable].elementType))
        x.copyStreamable(elementType, ptypes.forall(_.required))
      }
      case _: PSet => {
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PSet].elementType))
        PSet(elementType, ptypes.forall(_.required))
      }
      case x: PStruct => {
        PStruct(ptypes.forall(_.required), x.fieldNames.map( fieldName =>
          fieldName -> getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PStruct].field(fieldName).typ))
        ):_*)
      }
      case x: PTuple => {
        PTuple( ptypes.forall(_.required), x._types.map( pTupleField =>
          getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PTuple]._types(pTupleField.index).typ))
        ):_*)
      }
      case _: PDict => {
        val keyType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PDict].keyType))
        val valueType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PDict].valueType))

        PDict(keyType, valueType, ptypes.forall(_.required))
      }
      case _:PInterval => {
        val pointType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PInterval].pointType))
        PInterval(pointType, ptypes.forall(_.required))
      }
      case _ => ptypes.head.setRequired(ptypes.forall(_.required))
    }
  }

  def apply(ir: IR, env: Env[PType]): Unit = {
    assert(ir._pType == null)

    ir._pType = ir match {
      case I32(_) => PInt32(true)
      case I64(_) => PInt64(true)
      case F32(_) => PFloat32(true)
      case F64(_) => PFloat64(true)
      case Str(_) => PString(true)
      case Literal(t, _) => PType.canonical(t, true)
      case True() | False() => PBoolean(true)
      case Cast(ir, t) => {
        InferPType(ir, env)
        PType.canonical(t, ir._pType.required)
      }
      case CastRename(ir, t) => {
        InferPType(ir, env)
        ir._pType.deepRename(t)
      }
      case NA(t) => {
        PType.canonical(t).deepInnerRequired(false)
      }
      case Die(_, t) => {
        PType.canonical(t).deepInnerRequired(true)
      }
      case IsNA(ir) => {
        InferPType(ir, env)
        PBoolean(true)
      }
      case Ref(name, _) => env.lookup(name)
      case MakeNDArray(data, shape, rowMajor) => {
        InferPType(data, env)
        InferPType(shape, env)
        InferPType(rowMajor, env)

        val nElem = shape._pType.asInstanceOf[PTuple].size

        PNDArray(coerce[PArray](data._pType).elementType.setRequired(true), nElem, data._pType.required && shape._pType.required)
      }
      case ArrayRange(start: IR, stop: IR, step: IR) => {
        InferPType(start, env)
        InferPType(stop, env)
        InferPType(step, env)

        assert(start._pType isOfType stop._pType)
        assert(start._pType isOfType step._pType)

        val allRequired = start._pType.required && stop._pType.required && step._pType.required
        PArray(start._pType.setRequired(true), allRequired)
      }
      case StreamRange(start: IR, stop: IR, step: IR) => {
        InferPType(start, env)
        InferPType(stop, env)
        InferPType(step, env)

        assert(start._pType isOfType stop._pType)
        assert(start._pType isOfType step._pType)

        val allRequired = start._pType.required && stop._pType.required && step._pType.required
        PArray(start._pType.setRequired(true), allRequired)
      }
      case ArrayLen(a: IR) => {
        InferPType(a, env)

        PInt32(a._pType.required)
      }
      case LowerBoundOnOrderedCollection(orderedCollection: IR, bound: IR, _) => {
        InferPType(orderedCollection, env)
        InferPType(bound, env)

        PInt32(orderedCollection._pType.required)
      }
      case Let(name, value, body) => {
        InferPType(value, env)
        InferPType(body, env.bind(name, value._pType))

        body._pType
      }
      case TailLoop(_, args, body) =>
        args.foreach { case (_, ir) => InferPType(ir, env) }
        InferPType(body, env.bind(args.map { case (n, ir) => n -> ir._pType }: _*))
        body._pType
      case Recur(_, args, typ) =>
        args.foreach { a => InferPType(a, env) }
        PType.canonical(typ)
      case ApplyBinaryPrimOp(op, l, r) => {
        InferPType(l, env)
        InferPType(r, env)

        val required = l._pType.required && r._pType.required
        val vType = BinaryOp.getReturnType(op, l._pType.virtualType, r._pType.virtualType).setRequired(required)

        PType.canonical(vType, vType.required)
      }
      case ApplyUnaryPrimOp(op, v) => {
        InferPType(v, env)
        PType.canonical(UnaryOp.getReturnType(op, v._pType.virtualType).setRequired(v._pType.required))
      }
      case ApplyComparisonOp(op, l, r) => {
        InferPType(l, env)
        InferPType(r, env)

        assert(l._pType isOfType r._pType)
        op match {
          case _: Compare => PInt32(l._pType.required && r._pType.required)
          case _ => PBoolean(l._pType.required && r._pType.required)
        }
      }
      case a: ApplyIR => {
        InferPType(a.explicitNode, env)
        a.explicitNode._pType
      }
      case a: AbstractApplyNode[_] => {
        val pTypes = a.args.map( i => {
          InferPType(i, env)
          i._pType
        })
        a.implementation.returnPType(pTypes, a.returnType)
      }
      case a@ApplySpecial(_, args, _) => {
        val pTypes = args.map( i => {
          InferPType(i, env)
          i._pType
        })
        a.implementation.returnPType(pTypes, a.returnType)
      }
      case ArrayRef(a, i, s) => {
        InferPType(a, env)
        InferPType(i, env)
        InferPType(s, env)
        assert(i._pType isOfType PInt32() )

        coerce[PStreamable](a._pType).elementType.setRequired(a._pType.required && i._pType.required)
      }
      case ArraySort(a, leftName, rightName, compare) => {
        InferPType(a, env)
        val et = coerce[PStreamable](a._pType).elementType

        InferPType(compare, env.bind(leftName -> et, rightName -> et))
        assert(compare._pType.isOfType(PBoolean()))

        PArray(et, a._pType.required)
      }
      case ToSet(a) => {
        InferPType(a, env)
        val et = coerce[PIterable](a._pType).elementType
        PSet(et, a._pType.required)
      }
      case ToDict(a) => {
        InferPType(a, env)
        val elt = coerce[PBaseStruct](coerce[PIterable](a._pType).elementType)
        // Dict key/value types don't depend on PIterable's requiredeness because we have an interface guarantee that
        // null PIterables are filtered out before dict construction
        val keyRequired = elt.types(0).required
        val valRequired =  elt.types(1).required
        PDict(elt.types(0).setRequired(keyRequired), elt.types(1).setRequired(valRequired), a._pType.required)
      }
      case ToArray(a) => {
        InferPType(a, env)
        val elt = coerce[PIterable](a._pType).elementType
        PArray(elt, a._pType.required)
      }
      case ToStream(a) => {
        InferPType(a, env)
        val elt = coerce[PIterable](a._pType).elementType
        PStream(elt, a._pType.required)
      }
      case GroupByKey(collection) => {
        InferPType(collection, env)
        val elt = coerce[PBaseStruct](coerce[PStreamable](collection._pType).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection._pType.required)
      }
      case ArrayMap(a, name, body) => {
        InferPType(a, env)
        InferPType(body, env.bind(name, a._pType.asInstanceOf[PArray].elementType))
        coerce[PStreamable](a._pType).copyStreamable(body._pType, a._pType.required)
      }
      case ArrayZip(as, names, body, _) =>
        as.foreach(InferPType(_, env))

        InferPType(body, env.bindIterable(names.zip(as.map(_._pType.asInstanceOf[PArray].elementType))))
        coerce[PStreamable](as.head._pType).copyStreamable(body._pType, as.forall(_._pType.required))
      case ArrayFilter(a, name, cond) => {
        InferPType(a, env)
        a._pType
      }
      case ArrayFlatMap(a, name, body) => {
        InferPType(a, env)
        InferPType(body, env.bind(name, a._pType.asInstanceOf[PArray].elementType))

        // Whether an array must return depends on a, but element requiredeness depends on body (null a elements elided)
        coerce[PStreamable](a._pType).copyStreamable(coerce[PIterable](body._pType).elementType, a._pType.required)
      }
      case ArrayFold(a, zero, accumName, valueName, body) => {
        InferPType(zero, env)

        InferPType(a, env)
        InferPType(body, env.bind(accumName -> zero._pType, valueName -> a._pType.asInstanceOf[PArray].elementType))
        assert(body._pType isOfType zero._pType)

        zero._pType.setRequired(body._pType.required)
      }
      case ArrayFold2(a, acc, valueName, seq, res) =>
        InferPType(a, env)
        acc.foreach { case (_, accIR) => InferPType(accIR, env) }

        val resEnv = env.bind(acc.map { case (name, accIR) => (name, accIR._pType)}: _*)
        val seqEnv = resEnv.bind(valueName -> a._pType.asInstanceOf[PArray].elementType)
        seq.foreach(InferPType(_, seqEnv))
        InferPType(res, resEnv)
        res._pType.setRequired(res._pType.required && a._pType.required)
      case ArrayScan(a, zero, accumName, valueName, body) => {
        InferPType(zero, env)

        InferPType(a, env)
        InferPType(body, env.bind(accumName -> zero._pType, valueName -> a._pType.asInstanceOf[PArray].elementType))
        assert(body._pType isOfType zero._pType)

        val elementPType = zero._pType.setRequired(body._pType.required && zero._pType.required)
        coerce[PStreamable](a._pType).copyStreamable(elementPType, a._pType.required)
      }
      case ArrayLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) => {
        InferPType(lIR, env)
        InferPType(rIR, env)
        val e = env.bind(lName -> lIR._pType.asInstanceOf[PArray].elementType, rName -> rIR._pType.asInstanceOf[PArray].elementType)

        InferPType(compare, e)
        InferPType(join, e)

        PArray(join._pType, lIR._pType.required)
      }
      case NDArrayShape(nd) => {
        InferPType(nd, env)
        PTuple(nd._pType.required, IndexedSeq.tabulate(nd._pType.asInstanceOf[PNDArray].nDims)(_ => PInt64(true)):_*)
      }
      case NDArrayReshape(nd, shape) => {
        InferPType(nd, env)
        InferPType(shape, env)

        PNDArray(coerce[PNDArray](nd._pType).elementType, shape._pType.asInstanceOf[PTuple].size, nd._pType.required)
      }
      case NDArrayMap(nd, name, body) => {
        InferPType(nd, env)
        val ndPType = nd._pType.asInstanceOf[PNDArray]
        InferPType(body, env.bind(name -> ndPType.elementType))

        PNDArray(body._pType, ndPType.nDims, nd._pType.required)
      }
      case NDArrayMap2(l, r, lName, rName, body) => {
        InferPType(l, env)
        InferPType(r, env)

        val lPType = l._pType.asInstanceOf[PNDArray]
        val rPType = r._pType.asInstanceOf[PNDArray]

        InferPType(body, env.bind(lName -> lPType.elementType, rName -> rPType.elementType))

        PNDArray(body._pType, lPType.nDims, l._pType.required || r._pType.required)
      }
      case NDArrayReindex(nd, indexExpr) => {
        InferPType(nd, env)

        PNDArray(coerce[PNDArray](nd._pType).elementType, indexExpr.length, nd._pType.required)
      }
      case NDArrayRef(nd, idxs) => {
        InferPType(nd, env)

        var allRequired = nd._pType.required
        val it = idxs.iterator
        while(it.hasNext) {
          val idxIR = it.next()

          InferPType(idxIR, env)

          assert(idxIR._pType.isOfType(PInt64()) || idxIR._pType.isOfType(PInt32()))

          if (allRequired == true && idxIR._pType.required == false) {
            allRequired = false
          }
        }

        coerce[PNDArray](nd._pType).elementType.setRequired(allRequired)
      }
      case NDArraySlice(nd, slices) => {
        InferPType(nd, env)
        InferPType(slices, env)

        val remainingDims = coerce[PTuple](slices._pType).types.filter(_.isInstanceOf[PTuple])

        PNDArray(coerce[PNDArray](nd._pType).elementType, remainingDims.length, remainingDims.forall(_.required))
      }
      case NDArrayMatMul(l, r) => {
        InferPType(l, env)
        InferPType(r, env)
        val lTyp = coerce[PNDArray](l._pType)
        val rTyp = coerce[PNDArray](r._pType)
        PNDArray(lTyp.elementType, TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), lTyp.required && rTyp.required)
      }
      case NDArrayQR(nd, mode) => {
        InferPType(nd, env)
        mode match {
          case "r" => PNDArray(PFloat64Required, 2)
          case "raw" => PTuple(PNDArray(PFloat64Required, 2), PNDArray(PFloat64Required, 1))
          case "reduced" | "complete" => PTuple(PNDArray(PFloat64Required, 2), PNDArray(PFloat64Required, 2))
        }
      }
      case MakeStruct(fields) => PStruct(true, fields.map {
        case (name, a) => {
          InferPType(a, env)

          (name, a._pType)
        }
      }: _*)
      case SelectFields(old, fields) => {
        InferPType(old, env)
        val tbs = coerce[PStruct](old._pType)
        tbs.select(fields.toFastIndexedSeq)._1
      }
      case InsertFields(old, fields, fieldOrder) => {
        InferPType(old, env)
        val tbs = coerce[PStruct](old._pType)

        val s = tbs.insertFields(fields.map(f =>  {
          InferPType(f._2, env)
          (f._1, f._2._pType)
        }))

        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      }
      case GetField(o, name) => {
        InferPType(o, env)
        val t = coerce[PStruct](o._pType)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      }
      case MakeTuple(values) => PTuple(true, values.map(v => {
          InferPType(v._2, env)
          v._2._pType
      }):_*)
      case MakeArray(irs, t) => {
        if (irs.length == 0) {
          PType.canonical(t, true).deepInnerRequired(true)
        } else {
          val elementTypes = irs.map { elt =>
            InferPType(elt, env)
            elt._pType
          }

          val inferredElementType = getNestedElementPTypes(elementTypes)

          PArray(inferredElementType, true)
        }
      }
      case GetTupleElement(o, idx) => {
        InferPType(o, env)
        val t = coerce[PTuple](o._pType)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      }
      case CollectDistributedArray(contexts, globals, contextsName, globalsName, body) => {
        InferPType(contexts, env)
        InferPType(globals, env)

        InferPType(body, env.bind(contextsName -> contexts._pType.asInstanceOf[PArray].elementType, globalsName -> globals._pType))
        PArray(body._pType, true)
      }
      case If(cond, cnsq, altr) => {
        InferPType(cond, env)
        InferPType(cnsq, env)
        InferPType(altr, env)

        assert(cond._pType isOfType PBoolean())

        val branchType = getNestedElementPTypes(IndexedSeq(cnsq._pType, altr._pType))

        branchType.setRequired(branchType.required && cond._pType.required)
      }
      case Coalesce(values) =>
        getNestedElementPTypes(values.map( theIR => {
          InferPType(theIR, env)
          theIR._pType
        }))
      case In(_, pType: PType) => pType
      case x if x.typ == TVoid => {
        x.children.map(c => InferPType(c.asInstanceOf[IR], env))
        PVoid
      }
      case x@ResultOp(_, _) =>  PType.canonical(x.typ)
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) => {
        InferPType(contextsIR, env)
        InferPType(globalsIR, env)
        InferPType(bodyIR, env.bind(contextsName -> contextsIR._pType, globalsName -> globalsIR._pType))

        PCanonicalArray(bodyIR._pType, contextsIR._pType.required)
      }
      case ReadPartition(rowIR, codecSpec, rowType) => {
        InferPType(rowIR, env)

        val child = codecSpec.buildDecoder(rowType)._1

        PStream(child, child.required)
      }
      case MakeStream(irs, t) => {
        if (irs.length == 0) {
          PType.canonical(t, true).deepInnerRequired(true)
        }

        PStream(getNestedElementPTypes(irs.map( theIR => {
          InferPType(theIR, env)
          theIR._pType
        })), true)
      }
      case _:AggLet | _:ArrayAgg | _:ArrayAggScan | _:RunAgg | _:RunAggScan | _:NDArrayAgg | _:AggFilter | _:AggExplode |
           _:AggGroupBy | _:AggArrayPerElement | _:ApplyAggOp | _:ApplyScanOp | _:AggStateValue => PType.canonical(ir.typ)
    }

    // Allow only requiredeness to diverge
    assert(ir._pType.virtualType isOfType ir.typ)
  }
}

