package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.TNDArray
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
    assert(ir._pType2 == null)

    ir._pType2 = ir match {
      case I32(_) => PInt32(true)
      case I64(_) => PInt64(true)
      case F32(_) => PFloat32(true)
      case F64(_) => PFloat64(true)
      case Str(_) => PString(true)
      case Literal(t, _) => PType.canonical(t, true)
      case True() | False() => PBoolean(true)
      case Void() => PVoid
      case Cast(ir, t) => {
        InferPType(ir, env)
        PType.canonical(t, ir._pType2.required)
      }
      case CastRename(ir, t) => {
        InferPType(ir, env)
        ir._pType2.deepRename(t)
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
      case Ref(name, t) => env.lookup(name)
      case MakeNDArray(data, shape, rowMajor) => {
        InferPType(data, env)
        InferPType(shape, env)
        InferPType(rowMajor, env)

        val nElem = shape._pType2.asInstanceOf[PTuple].size

        PNDArray(coerce[PArray](data._pType2).elementType.setRequired(true), nElem, data._pType2.required && shape._pType2.required)
      }
      case ArrayRange(start: IR, stop: IR, step: IR) => {
        InferPType(start, env)
        InferPType(stop, env)
        InferPType(step, env)

        assert(start._pType2 isOfType stop._pType2)
        assert(start._pType2 isOfType step._pType2)

        val allRequired = start._pType2.required && stop._pType2.required && step._pType2.required
        PArray(start._pType2.setRequired(true), allRequired)
      }
      case StreamRange(start: IR, stop: IR, step: IR) => {
        InferPType(start, env)
        InferPType(stop, env)
        InferPType(step, env)

        assert(start._pType2 isOfType stop._pType2)
        assert(start._pType2 isOfType step._pType2)

        val allRequired = start._pType2.required && stop._pType2.required && step._pType2.required
        PArray(start._pType2.setRequired(true), allRequired)
      }
      case ArrayLen(a: IR) => {
        InferPType(a, env)

        PInt32(a._pType2.required)
      }
      case LowerBoundOnOrderedCollection(orderedCollection: IR, bound: IR, _) => {
        InferPType(orderedCollection, env)
        InferPType(bound, env)

        PInt32(orderedCollection._pType2.required)
      }
      case _: ArrayFor => PVoid
      case _: Begin => PVoid
      case Let(name, value, body) => {
        InferPType(value, env)
        InferPType(body, env.bind(name, value._pType2))

        body._pType2
      }
      case TailLoop(_, args, body) =>
        args.foreach { case (_, ir) => InferPType(ir, env) }
        InferPType(body, env.bind(args.map { case (n, ir) => n -> ir._pType2 }: _*))
        body._pType2
      case Recur(_, args, typ) =>
        // FIXME: This may be difficult to infer properly from a bottom-up pass.
        args.foreach { a => InferPType(a, env) }
        PType.canonical(typ)

      case ApplyBinaryPrimOp(op, l, r) => {
          InferPType(l, env)
          InferPType(r, env)

          val required = l._pType2.required && r._pType2.required
          val vType = BinaryOp.getReturnType(op, l._pType2.virtualType, r._pType2.virtualType).setRequired(required)

          PType.canonical(vType, vType.required)
      }
      case ApplyUnaryPrimOp(op, v) => {
        InferPType(v, env)
        PType.canonical(UnaryOp.getReturnType(op, v._pType2.virtualType).setRequired(v._pType2.required))
      }
      case ApplyComparisonOp(op, l, r) => {
        InferPType(l, env)
        InferPType(r, env)

        assert(l._pType2 isOfType r._pType2)
        op match {
          case _: Compare => PInt32(l._pType2.required && r._pType2.required)
          case _ => PBoolean(l._pType2.required && r._pType2.required)
        }
      }
      case a: ApplyIR => {
        InferPType(a.explicitNode, env)
        a.explicitNode._pType2
      }
      case a: AbstractApplyNode[_] => {
        val pTypes = a.args.map( i => {
          InferPType(i, env)
          i._pType2
        })
        a.implementation.returnPType(pTypes, a.returnType)
      }
      case a@ApplySpecial(_, args, _) => {
        val pTypes = args.map( i => {
          InferPType(i, env)
          i._pType2
        })
        a.implementation.returnPType(pTypes, a.returnType)
      }
      case ArrayRef(a, i, s) => {
        InferPType(a, env)
        InferPType(i, env)
        InferPType(s, env)
        assert(i._pType2 isOfType PInt32() )

        coerce[PStreamable](a._pType2).elementType.setRequired(a._pType2.required && i._pType2.required)
      }
      case ArraySort(a, leftName, rightName, compare) => {
        InferPType(a, env)
        val et = coerce[PStreamable](a._pType2).elementType

        InferPType(compare, env.bind(leftName -> et, rightName -> et))
        assert(compare._pType2.isOfType(PBoolean()))

        PArray(et, a._pType2.required)
      }
      case ToSet(a) => {
        InferPType(a, env)
        val et = coerce[PIterable](a._pType2).elementType
        PSet(et, a._pType2.required)
      }
      case ToDict(a) => {
        InferPType(a, env)
        val elt = coerce[PBaseStruct](coerce[PIterable](a._pType2).elementType)
        // Dict key/value types don't depend on PIterable's requiredeness because we have an interface guarantee that
        // null PIterables are filtered out before dict construction
        val keyRequired = elt.types(0).required
        val valRequired =  elt.types(1).required
        PDict(elt.types(0).setRequired(keyRequired), elt.types(1).setRequired(valRequired), a._pType2.required)
      }
      case ToArray(a) => {
        InferPType(a, env)
        val elt = coerce[PIterable](a._pType2).elementType
        PArray(elt, a._pType2.required)
      }
      case ToStream(a) => {
        InferPType(a, env)
        val elt = coerce[PIterable](a._pType2).elementType
        PStream(elt, a._pType2.required)
      }
      case GroupByKey(collection) => {
        InferPType(collection, env)
        val elt = coerce[PBaseStruct](coerce[PStreamable](collection._pType2).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection._pType2.required)
      }
      case ArrayMap(a, name, body) => {
        InferPType(a, env)
        InferPType(body, env.bind(name, a._pType2.asInstanceOf[PArray].elementType))
        coerce[PStreamable](a._pType2).copyStreamable(body._pType2, body._pType2.required)
      }
      case ArrayZip(as, names, body, _) =>
        as.foreach(a => InferPType(a, env))

        InferPType(body, env.bindIterable(names.zip(as.map(_._pType2.asInstanceOf[PArray].elementType))))
        coerce[PStreamable](as.head._pType2).copyStreamable(body._pType2, as.forall(_._pType2.required))
      case ArrayFilter(a, name, cond) => {
        InferPType(a, env)
        a._pType2
      }
      case ArrayFlatMap(a, name, body) => {
        InferPType(a, env)
        InferPType(body, env.bind(name, a._pType2.asInstanceOf[PArray].elementType))

        // Whether an array must return depends on a, but element requiredeness depends on body (null a elements elided)
        coerce[PStreamable](a._pType2).copyStreamable(coerce[PIterable](body._pType2).elementType, a._pType2.required)
      }
      case ArrayFold(a, zero, accumName, valueName, body) => {
        InferPType(zero, env)

        InferPType(a, env)
        InferPType(body, env.bind(accumName -> zero._pType2, valueName -> a._pType2.asInstanceOf[PArray].elementType))
        assert(body._pType2 isOfType zero._pType2)

        zero._pType2.setRequired(body._pType2.required)
      }
      case ArrayFold2(a, acc, valueName, seq, res) =>
        InferPType(a, env)
        acc.foreach { case (_, accIR) => InferPType(accIR, env) }

        val resEnv = env.bind(acc.map { case (name, accIR) => (name, accIR._pType2)}: _*)
        val seqEnv = resEnv.bind(valueName -> a._pType2.asInstanceOf[PArray].elementType)
        seq.foreach(InferPType(_, seqEnv))
        InferPType(res, resEnv)
        res._pType2.setRequired(res._pType2.required && a._pType2.required)
      case ArrayScan(a, zero, accumName, valueName, body) => {
        InferPType(zero, env)

        InferPType(a, env)
        InferPType(body, env.bind(accumName -> zero._pType2, valueName -> a._pType2.asInstanceOf[PArray].elementType))
        assert(body._pType2 isOfType zero._pType2)

        val elementPType = zero._pType2.setRequired(body._pType2.required && zero._pType2.required)
        coerce[PStreamable](a._pType2).copyStreamable(elementPType, a._pType2.required)
      }
      case ArrayLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) => {
        InferPType(lIR, env)
        InferPType(rIR, env)

        InferPType(join, env.bind(lName -> lIR._pType2.asInstanceOf[PArray].elementType, rName -> rIR._pType2.asInstanceOf[PArray].elementType))

        PArray(join._pType2, lIR._pType2.required)
      }
      case NDArrayShape(nd) => {
        InferPType(nd, env)
        PTuple(nd._pType2.required, IndexedSeq.tabulate(nd._pType2.asInstanceOf[PNDArray].nDims)(_ => PInt64(true)):_*)
      }
      case NDArrayReshape(nd, shape) => {
        InferPType(nd, env)
        InferPType(shape, env)

        PNDArray(coerce[PNDArray](nd._pType2).elementType, shape._pType2.asInstanceOf[PTuple].size, nd._pType2.required)
      }
      case NDArrayMap(nd, name, body) => {
        InferPType(nd, env)
        InferPType(body, env.bind(name, nd._pType2))

        PNDArray(body._pType2, coerce[PNDArray](nd._pType2).nDims, nd._pType2.required)
      }
      case NDArrayMap2(l, r, lName, rName, body) => {
        InferPType(l, env)
        InferPType(body, env)

        PNDArray(body._pType2, coerce[PNDArray](l._pType2).nDims, l._pType2.required)
      }
      case NDArrayReindex(nd, indexExpr) => {
        InferPType(nd, env)

        PNDArray(coerce[PNDArray](nd._pType2).elementType, indexExpr.length, nd._pType2.required)
      }
      case NDArrayRef(nd, idxs) => {
        InferPType(nd, env)

        var allRequired = nd._pType2.required
        val it = idxs.iterator
        while(it.hasNext) {
          val idxIR = it.next()

          InferPType(idxIR, env)

          assert(idxIR._pType2.isOfType(PInt64()) || idxIR._pType2.isOfType(PInt32()))

          if (allRequired == true && idxIR._pType2.required == false) {
            allRequired = false
          }
        }

        coerce[PNDArray](nd._pType2).elementType.setRequired(allRequired)
      }
      case NDArraySlice(nd, slices) => {
        InferPType(nd, env)
        InferPType(slices, env)

        val remainingDims = coerce[PTuple](slices._pType2).types.filter(_.isInstanceOf[PTuple])

        PNDArray(coerce[PNDArray](nd._pType2).elementType, remainingDims.length, remainingDims.forall(_.required))
      }
      case NDArrayMatMul(l, r) => {
        InferPType(l, env)
        InferPType(r, env)
        val lTyp = coerce[PNDArray](l._pType2)
        val rTyp = coerce[PNDArray](r._pType2)
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
      case NDArrayWrite(_, _) => PVoid
      case MakeStruct(fields) => PStruct(true, fields.map {
        case (name, a) => {
          InferPType(a, env)

          (name, a._pType2)
        }
      }: _*)
      case SelectFields(old, fields) => {
        InferPType(old, env)
        val tbs = coerce[PStruct](old._pType2)
        tbs.select(fields.toFastIndexedSeq)._1
      }
      case InsertFields(old, fields, fieldOrder) => {
        InferPType(old, env)
        val tbs = coerce[PStruct](old._pType2)

        val s = tbs.insertFields(fields.map(f =>  {
          InferPType(f._2, env)
          (f._1, f._2._pType2)
        }))

        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      }
      case GetField(o, name) => {
        InferPType(o, env)
        val t = coerce[PStruct](o._pType2)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      }
      case MakeTuple(values) => PTuple(true, values.map(v => {
          InferPType(v._2, env)
          v._2._pType2
      }):_*)
      case MakeArray(irs, t) => {
        if (irs.length == 0) {
          PType.canonical(t, true).deepInnerRequired(true)
        } else {
          val elementTypes = irs.map { elt =>
            InferPType(elt, env)
            elt._pType2
          }

          val inferredElementType = getNestedElementPTypes(elementTypes)

          PArray(inferredElementType, true)
        }
      }
      case GetTupleElement(o, idx) => {
        InferPType(o, env)
        val t = coerce[PTuple](o._pType2)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      }
      case CollectDistributedArray(contexts, globals, contextsName, globalsName, body) => {
        InferPType(contexts, env)
        InferPType(globals, env)

        InferPType(body, env.bind(contextsName -> contexts._pType2.asInstanceOf[PArray].elementType, globalsName -> globals._pType2))
        PArray(body._pType2, body._pType2.required)
      }
      case If(cond, cnsq, altr) => {
        InferPType(cond, env)
        InferPType(cnsq, env)
        InferPType(altr, env)

        assert(cond._pType2 isOfType PBoolean())

        val branchType = getNestedElementPTypes(IndexedSeq(cnsq._pType2, altr._pType2))

        branchType.setRequired(branchType.required && cond._pType2.required)
      }

      case Coalesce(values) =>
        getNestedElementPTypes(values.map( theIR => {
          InferPType(theIR, env)
          theIR._pType2
        }))
      case In(_, pType: PType) => pType
      case NDArrayWrite(_, _) => PVoid
      case _: InitOp => PVoid
      case _: SeqOp => PVoid
      case _: CombOp => PVoid
      case x@ResultOp(_, _) =>
        PType.canonical(InferType(x))
      case _: SerializeAggs => PVoid
      case _: DeserializeAggs => PVoid
      case _: Begin => PVoid
      case _: TableWrite => PVoid
      case _: TableMultiWrite => PVoid
      case _: MatrixWrite => PVoid
      case _: MatrixMultiWrite => PVoid
      case _: BlockMatrixWrite => PVoid
      case _: BlockMatrixMultiWrite => PVoid
      case TableGetGlobals(child) => PType.canonical(child.typ.globalType)
      case x@TableCollect(_) =>
        PType.canonical(InferType(x))
      case TableToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case MatrixToValueApply(child, function) => PType.canonical(function.typ(child.typ))
      case BlockMatrixToValueApply(child, function) =>
        PType.canonical(function.typ(child.typ))
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) => {
        InferPType(contextsIR, env)
        InferPType(globalsIR, env)
        InferPType(bodyIR, env.bind(contextsName -> contextsIR._pType2, globalsName -> globalsIR._pType2))

        PCanonicalArray(bodyIR._pType2, contextsIR._pType2.required)
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
          theIR._pType2
        })), true)
      }
      case AggLet(name, valueIR, bodyIR, _) => {
        InferPType(valueIR, env)
        InferPType(bodyIR, env.bind(name -> valueIR._pType2))
        bodyIR._pType2
      }
      case ArrayAgg(a, name, queryIR) => {
        InferPType(a, env)
        InferPType(queryIR, env.bind(name -> a._pType2.asInstanceOf[PArray].elementType))
        queryIR._pType2
      }
      case ArrayAggScan(a, name, queryIR)  => {
        InferPType(a, env)
        InferPType(queryIR, env.bind(name -> a._pType2.asInstanceOf[PArray].elementType))
        PCanonicalArray(queryIR._pType2, queryIR._pType2.required)
      }
      case RunAgg(bodyIR, resultIR, _) => {
        InferPType(bodyIR, env)
        InferPType(resultIR, env)
        resultIR._pType2
      }
      case RunAggScan(arrayIR, name, initOpIR, seqOpIR, resultIR, _) => {
        InferPType(arrayIR, env)
        val e = env.bind(name -> arrayIR._pType2)

        InferPType(initOpIR, e)
        InferPType(seqOpIR, e)
        InferPType(resultIR, e)

        PCanonicalArray(resultIR._pType2, resultIR._pType2.required)
      }
      case NDArrayAgg(ndIR, axes) => {
        InferPType(ndIR, env)
        val childPType = ndIR._pType2.asInstanceOf[PNDArray]
        PCanonicalNDArray(childPType.elementType, childPType.nDims - axes.length, childPType.required)
      }
      case AggFilter(condIR, aggIR, _) => {
        InferPType(condIR, env)
        assert(condIR._pType2.isOfType(PBoolean()))

        InferPType(aggIR, env)
        aggIR._pType2
      }
      case AggExplode(arrayIR, name, aggBody, _) => {
        InferPType(arrayIR, env)
        InferPType(aggBody, env.bind(name -> arrayIR._pType2.asInstanceOf[PIterable].elementType))
        aggBody._pType2
      }
      case AggGroupBy(keyIR, aggIR, _) => {
        InferPType(keyIR, env)
        InferPType(aggIR, env)
        PCanonicalDict(keyIR._pType2, aggIR._pType2, keyIR._pType2.required && aggIR._pType2.required)
      }
      case AggArrayPerElement(a: IR, elementName, indexName, aggBody, knownLength, _) => {
        InferPType(a, env)

        knownLength match {
          case Some(kIR) => InferPType(kIR, env)
          case None =>
        }

        InferPType(aggBody, env.bind(elementName -> a._pType2.asInstanceOf[PArray].elementType))
        PArray(aggBody._pType2, aggBody._pType2.required)
      }
        // TODO: need to bind name for initOpArgs to find
      case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) => {
        var i = 0
        val iPTypes = initOpArgs.map(i => {
          InferPType(i, env)
          i._pType2
        })
        val sPTypes = seqOpArgs.map(s => {
          InferPType(s, env)
          s._pType2
        })

        aggSig.toPhysical(initOpTypes = iPTypes, seqOpTypes = sPTypes).returnType
      }
      case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) => {
        val iPTypes = initOpArgs.map(i => {
          InferPType(i, env)
          i._pType2
        })
        val sPTypes = seqOpArgs.map(s => {
          InferPType(s, env)
          s._pType2
        })

        aggSig.toPhysical(initOpTypes = iPTypes, seqOpTypes = sPTypes).returnType
      }
      case AggStateValue(_, _) => PBinary(true)
      case TableCount(_) => PInt64(true)
      case TableAggregate(_, query) => {
        InferPType(query, env)
        query._pType2
      }
      case MatrixAggregate(_, query) => {
        InferPType(query, env)
        query._pType2
      }
      case _ => throw new Exception("Node not supported")
    }

    // Allow only requiredeness to diverge
    assert(ir._pType2.virtualType isOfType ir.typ)
  }
}

