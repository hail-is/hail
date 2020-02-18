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
      case x: PStreamable =>
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PStreamable].elementType))
        x.copyStreamable(elementType, ptypes.forall(_.required))
      case _: PSet =>
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PSet].elementType))
        PSet(elementType, ptypes.forall(_.required))
      case x: PStruct =>
        PStruct(ptypes.forall(_.required), x.fieldNames.map(fieldName =>
          fieldName -> getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PStruct].field(fieldName).typ))
        ): _*)
      case x: PTuple =>
        PTuple(ptypes.forall(_.required), x._types.map(pTupleField =>
          getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PTuple]._types(pTupleField.index).typ))
        ): _*)
      case _: PDict =>
        val keyType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PDict].keyType))
        val valueType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PDict].valueType))

        PDict(keyType, valueType, ptypes.forall(_.required))
      case _: PInterval =>
        val pointType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PInterval].pointType))
        PInterval(pointType, ptypes.forall(_.required))
      case _ => ptypes.head.setRequired(ptypes.forall(_.required))
    }
  }

  def apply(ir: IR, env: Env[PType]): Unit = {
    assert(ir._pType2 == null)

    def infer(ir: IR, env: Env[PType] = env): Unit = apply(ir, env)

    ir._pType2 = ir match {
      case I32(_) => PInt32(true)
      case I64(_) => PInt64(true)
      case F32(_) => PFloat32(true)
      case F64(_) => PFloat64(true)
      case Str(_) => PString(true)
      case Literal(t, _) => PType.canonical(t, true)
      case True() | False() => PBoolean(true)
      case Cast(ir, t) =>
        infer(ir)
        PType.canonical(t, ir.pType2.required)
      case CastRename(ir, t) =>
        infer(ir)
        ir._pType2.deepRename(t)
      case NA(t) =>
        PType.canonical(t).deepInnerRequired(false)
      case Die(_, t) =>
        PType.canonical(t).deepInnerRequired(true)
      case IsNA(ir) =>
        infer(ir)
        PBoolean(true)
      case Ref(name, _) => env.lookup(name)
      case MakeNDArray(data, shape, rowMajor) =>
        infer(data)
        infer(shape)
        infer(rowMajor)

        val nElem = shape.pType2.asInstanceOf[PTuple].size

        PNDArray(coerce[PArray](data.pType2).elementType.setRequired(true), nElem, data.pType2.required && shape.pType2.required)
      case ArrayRange(start: IR, stop: IR, step: IR) =>
        infer(start)
        infer(stop)
        infer(step)

        assert(start.pType2 isOfType stop.pType2)
        assert(start.pType2 isOfType step.pType2)

        val allRequired = start.pType2.required && stop.pType2.required && step.pType2.required
        PArray(start.pType2.setRequired(true), allRequired)
      case StreamRange(start: IR, stop: IR, step: IR) =>
        infer(start)
        infer(stop)
        infer(step)

        assert(start.pType2 isOfType stop.pType2)
        assert(start.pType2 isOfType step.pType2)

        val allRequired = start.pType2.required && stop.pType2.required && step.pType2.required
        PArray(start.pType2.setRequired(true), allRequired)
      case ArrayLen(a: IR) =>
        infer(a)

        PInt32(a.pType2.required)
      case LowerBoundOnOrderedCollection(orderedCollection: IR, bound: IR, _) =>
        infer(orderedCollection)
        infer(bound)

        PInt32(orderedCollection.pType2.required)
      case Let(name, value, body) =>
        infer(value)
        infer(body, env.bind(name, value.pType2))

        body.pType2
      case TailLoop(_, args, body) =>
        args.foreach { case (_, ir) => infer(ir) }
        infer(body, env.bind(args.map { case (n, ir) => n -> ir.pType2 }: _*))
        body.pType2
      case Recur(_, args, typ) =>
        args.foreach { a => infer(a) }
        PType.canonical(typ)
      case ApplyBinaryPrimOp(op, l, r) =>
        infer(l)
        infer(r)

        val required = l.pType2.required && r.pType2.required
        val vType = BinaryOp.getReturnType(op, l.pType2.virtualType, r.pType2.virtualType).setRequired(required)

        PType.canonical(vType, vType.required)
      case ApplyUnaryPrimOp(op, v) =>
        infer(v)
        PType.canonical(UnaryOp.getReturnType(op, v.pType2.virtualType).setRequired(v.pType2.required))
      case ApplyComparisonOp(op, l, r) =>
        infer(l)
        infer(r)

        assert(l.pType2 isOfType r.pType2)
        op match {
          case _: Compare => PInt32(l.pType2.required && r.pType2.required)
          case _ => PBoolean(l.pType2.required && r.pType2.required)
        }
      case a: ApplyIR =>
        infer(a.explicitNode)
        a.explicitNode.pType2
      case a: AbstractApplyNode[_] =>
        val pTypes = a.args.map(i => {
          infer(i)
          i.pType2
        })
        a.implementation.returnPType(pTypes, a.returnType)
      case a@ApplySpecial(_, args, _) =>
        val pTypes = args.map(i => {
          infer(i)
          i.pType2
        })
        a.implementation.returnPType(pTypes, a.returnType)
      case ArrayRef(a, i, s) =>
        infer(a)
        infer(i)
        infer(s)
        assert(i.pType2 isOfType PInt32())

        coerce[PStreamable](a.pType2).elementType.setRequired(a.pType2.required && i.pType2.required)
      case ArraySort(a, leftName, rightName, compare) =>
        infer(a)
        val et = coerce[PStreamable](a.pType2).elementType

        infer(compare, env.bind(leftName -> et, rightName -> et))
        assert(compare.pType2.isOfType(PBoolean()))

        PArray(et, a.pType2.required)
      case ToSet(a) =>
        infer(a)
        val et = coerce[PIterable](a.pType2).elementType
        PSet(et, a.pType2.required)
      case ToDict(a) =>
        infer(a)
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType2).elementType)
        // Dict key/value types don't depend on PIterable's requiredeness because we have an interface guarantee that
        // null PIterables are filtered out before dict construction
        val keyRequired = elt.types(0).required
        val valRequired = elt.types(1).required
        PDict(elt.types(0).setRequired(keyRequired), elt.types(1).setRequired(valRequired), a.pType2.required)
      case ToArray(a) =>
        infer(a)
        val elt = coerce[PIterable](a.pType2).elementType
        PArray(elt, a.pType2.required)
      case ToStream(a) =>
        infer(a)
        val elt = coerce[PIterable](a.pType2).elementType
        PStream(elt, a.pType2.required)
      case GroupByKey(collection) =>
        infer(collection)
        val elt = coerce[PBaseStruct](coerce[PStreamable](collection.pType2).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection.pType2.required)
      case ArrayMap(a, name, body) =>
        infer(a)
        infer(body, env.bind(name, a.pType2.asInstanceOf[PArray].elementType))
        coerce[PStreamable](a.pType2).copyStreamable(body.pType2, a.pType2.required)
      case ArrayZip(as, names, body, _) =>
        as.foreach(infer(_))

        infer(body, env.bindIterable(names.zip(as.map(_.pType2.asInstanceOf[PArray].elementType))))
        coerce[PStreamable](as.head.pType2).copyStreamable(body.pType2, as.forall(_.pType2.required))
      case ArrayFilter(a, name, cond) =>
        infer(a)
        a.pType2
      case ArrayFlatMap(a, name, body) =>
        infer(a)
        infer(body, env.bind(name, a.pType2.asInstanceOf[PArray].elementType))

        // Whether an array must return depends on a, but element requiredeness depends on body (null a elements elided)
        coerce[PStreamable](a.pType2).copyStreamable(coerce[PIterable](body.pType2).elementType, a.pType2.required)
      case ArrayFold(a, zero, accumName, valueName, body) =>
        infer(zero)

        infer(a)
        infer(body, env.bind(accumName -> zero.pType2, valueName -> a.pType2.asInstanceOf[PArray].elementType))
        assert(body.pType2 isOfType zero.pType2)

        zero.pType2.setRequired(body.pType2.required)
      case ArrayFold2(a, acc, valueName, seq, res) =>
        infer(a)
        acc.foreach { case (_, accIR) => infer(accIR) }
        val resEnv = env.bind(acc.map { case (name, accIR) => (name, accIR.pType2) }: _*)
        val seqEnv = resEnv.bind(valueName -> a.pType2.asInstanceOf[PArray].elementType)
        seq.foreach(infer(_, seqEnv))
        infer(res, resEnv)
        res.pType2.setRequired(res.pType2.required && a.pType2.required)
      case ArrayScan(a, zero, accumName, valueName, body) =>
        infer(zero)

        infer(a)
        infer(body, env.bind(accumName -> zero.pType2, valueName -> a.pType2.asInstanceOf[PArray].elementType))
        assert(body.pType2 isOfType zero.pType2)

        val elementPType = zero.pType2.setRequired(body.pType2.required && zero.pType2.required)
        coerce[PStreamable](a.pType2).copyStreamable(elementPType, a.pType2.required)
      case ArrayLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) =>
        infer(lIR)
        infer(rIR)
        val e = env.bind(lName -> lIR.pType2.asInstanceOf[PArray].elementType, rName -> rIR.pType2.asInstanceOf[PArray].elementType)

        infer(compare, e)
        infer(join, e)

        PArray(join.pType2, lIR.pType2.required)
      case NDArrayShape(nd) =>
        infer(nd)
        PTuple(nd.pType2.required, IndexedSeq.tabulate(nd.pType2.asInstanceOf[PNDArray].nDims)(_ => PInt64(true)): _*)
      case NDArrayReshape(nd, shape) =>
        infer(nd)
        infer(shape)

        PNDArray(coerce[PNDArray](nd.pType2).elementType, shape.pType2.asInstanceOf[PTuple].size, nd.pType2.required)
      case NDArrayConcat(nds, _) =>
        infer(nds)
        val ndtyp = coerce[PNDArray](coerce[PStreamable](nds.pType2).elementType)
        ndtyp
      case NDArrayMap(nd, name, body) =>
        infer(nd)
        val ndPType = nd.pType2.asInstanceOf[PNDArray]
        infer(body, env.bind(name -> ndPType.elementType))

        PNDArray(body.pType2, ndPType.nDims, nd.pType2.required)
      case NDArrayMap2(l, r, lName, rName, body) =>
        infer(l)
        infer(r)

        val lPType = l.pType2.asInstanceOf[PNDArray]
        val rPType = r.pType2.asInstanceOf[PNDArray]

        InferPType(body, env.bind(lName -> lPType.elementType, rName -> rPType.elementType))

        PNDArray(body.pType2, lPType.nDims, l.pType2.required || r.pType2.required)
      case NDArrayReindex(nd, indexExpr) =>
        infer(nd)

        PNDArray(coerce[PNDArray](nd.pType2).elementType, indexExpr.length, nd.pType2.required)
      case NDArrayRef(nd, idxs) =>
        infer(nd)

        var allRequired = nd.pType2.required
        val it = idxs.iterator
        while (it.hasNext) {
          val idxIR = it.next()
          infer(idxIR)
          assert(idxIR.pType2.isOfType(PInt64()) || idxIR.pType2.isOfType(PInt32()))
          if (allRequired && !idxIR.pType2.required) {
            allRequired = false
          }
        }

        coerce[PNDArray](nd.pType2).elementType.setRequired(allRequired)
      case NDArraySlice(nd, slices) =>
        infer(nd)
        infer(slices)
        val remainingDims = coerce[PTuple](slices.pType2).types.filter(_.isInstanceOf[PTuple])
        PNDArray(coerce[PNDArray](nd.pType2).elementType, remainingDims.length, remainingDims.forall(_.required))
      case NDArrayMatMul(l, r) =>
        infer(l)
        infer(r)
        val lTyp = coerce[PNDArray](l.pType2)
        val rTyp = coerce[PNDArray](r.pType2)
        PNDArray(lTyp.elementType, TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), lTyp.required && rTyp.required)
      case NDArrayQR(nd, mode) =>
        infer(nd)
        mode match {
          case "r" => PNDArray(PFloat64Required, 2)
          case "raw" => PTuple(PNDArray(PFloat64Required, 2), PNDArray(PFloat64Required, 1))
          case "reduced" | "complete" => PTuple(PNDArray(PFloat64Required, 2), PNDArray(PFloat64Required, 2))
        }
      case MakeStruct(fields) =>
        PStruct(true, fields.map { case (name, a) =>
          infer(a)
          (name, a.pType2)
        }: _ *)
      case SelectFields(old, fields) =>
        infer(old)
        val tbs = coerce[PStruct](old.pType2)
        tbs.select(fields.toFastIndexedSeq)._1
      case InsertFields(old, fields, fieldOrder) =>
        infer(old)
        val tbs = coerce[PStruct](old.pType2)

        val s = tbs.insertFields(fields.map(f => {
          infer(f._2)
          (f._1, f._2.pType2)
        }))

        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      case GetField(o, name) =>
        infer(o)
        val t = coerce[PStruct](o.pType2)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      case MakeTuple(values) => PTuple(true, values.map(v => {
        infer(v._2)
        v._2.pType2
      }): _*)
      case MakeArray(irs, t) =>
        if (irs.isEmpty) {
          PType.canonical(t, true).deepInnerRequired(true)
        } else {
          val elementTypes = irs.map { elt =>
            infer(elt)
            elt.pType2
          }

          val inferredElementType = getNestedElementPTypes(elementTypes)
          PArray(inferredElementType, true)
        }
      case GetTupleElement(o, idx) =>
        infer(o)
        val t = coerce[PTuple](o.pType2)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      case If(cond, cnsq, altr) =>
        infer(cond)
        infer(cnsq)
        infer(altr)

        assert(cond.pType2 isOfType PBoolean())

        val branchType = getNestedElementPTypes(IndexedSeq(cnsq.pType2, altr.pType2))

        branchType.setRequired(branchType.required && cond.pType2.required)
      case Coalesce(values) =>
        getNestedElementPTypes(values.map(theIR => {
          infer(theIR)
          theIR._pType2
        }))
      case In(_, pType: PType) => pType

      case x if x.typ == TVoid =>
        x.children.foreach(c => infer(c.asInstanceOf[IR]))
        PVoid
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        infer(contextsIR)
        infer(globalsIR)
        infer(bodyIR, env.bind(contextsName -> coerce[PStreamable](contextsIR._pType2).elementType, globalsName -> globalsIR._pType2))

        PCanonicalArray(bodyIR._pType2, contextsIR._pType2.required)
      case ReadPartition(rowIR, codecSpec, rowType) =>
        infer(rowIR)

        val child = codecSpec.buildDecoder(rowType)._1

        PStream(child, child.required)
      case MakeStream(irs, t) =>
        if (irs.isEmpty) {
          PType.canonical(t, true).deepInnerRequired(true)
        }

        PStream(getNestedElementPTypes(irs.map(theIR => {
          infer(theIR)
          theIR._pType2
        })), true)
      case _: AggLet | _: ArrayAgg | _: ArrayAggScan | _: RunAgg | _: RunAggScan | _: NDArrayAgg | _: AggFilter | _: AggExplode |
           _: AggGroupBy | _: AggArrayPerElement | _: ApplyAggOp | _: ApplyScanOp | _: AggStateValue => PType.canonical(ir.typ)
    }

    // Allow only requiredeness to diverge
    assert(ir.pType2.virtualType isOfType ir.typ)
  }
}

