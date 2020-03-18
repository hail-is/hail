package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TNDArray, TVoid}
import is.hail.utils._

object InferPType {

  def clearPTypes(x: BaseIR): Unit = {
    x match {
      case x: IR =>
        x._pType = null
      case _ =>
    }
    x.children.foreach(clearPTypes)
  }

  // does not unify physical arg types if multiple nested seq/init ops appear; instead takes the first. The emitter checks equality.
  def computePhysicalAgg(virt: AggStateSignature, initsAB: ArrayBuilder[RecursiveArrayBuilderElement[InitOp]],
    seqAB: ArrayBuilder[RecursiveArrayBuilderElement[SeqOp]]): AggStatePhysicalSignature = {
    val inits = initsAB.result()
    val seqs = seqAB.result()
    assert(inits.nonEmpty)
    assert(seqs.nonEmpty)
    virt.default match {
      case AggElementsLengthCheck() =>
        assert(inits.length == 1)
        assert(seqs.length == 2)

        val iHead = inits.find(_.value.op == AggElementsLengthCheck()).get
        val iNested = iHead.nested.get
        val iHeadArgTypes = iHead.value.args.map(_.pType)

        val sLCHead = seqs.find(_.value.op == AggElementsLengthCheck()).get
        val sLCArgTypes = sLCHead.value.args.map(_.pType)
        val sAEHead = seqs.find(_.value.op == AggElements()).get
        val sNested = sAEHead.nested.get
        val sHeadArgTypes = sAEHead.value.args.map(_.pType)

        val vNested = virt.nested.get.toArray

        val nested = vNested.indices.map { i => computePhysicalAgg(vNested(i), iNested(i), sNested(i)) }
        AggStatePhysicalSignature(Map(
          AggElementsLengthCheck() -> PhysicalAggSignature(AggElementsLengthCheck(), iHeadArgTypes, sLCArgTypes),
          AggElements() -> PhysicalAggSignature(AggElements(), FastIndexedSeq(), sHeadArgTypes)
        ), AggElementsLengthCheck(), Some(nested))

      case Group() =>
        assert(inits.length == 1)
        assert(seqs.length == 1)
        val iHead = inits.head
        val iNested = iHead.nested.get
        val iHeadArgTypes = iHead.value.args.map(_.pType)

        val sHead = seqs.head
        val sNested = sHead.nested.get
        val sHeadArgTypes = sHead.value.args.map(_.pType)

        val vNested = virt.nested.get.toArray

        val nested = vNested.indices.map { i => computePhysicalAgg(vNested(i), iNested(i), sNested(i)) }
        val psig = PhysicalAggSignature(Group(), iHeadArgTypes, sHeadArgTypes)
        AggStatePhysicalSignature(Map(Group() -> psig), Group(), Some(nested))

      case _ =>
        assert(inits.forall(_.nested.isEmpty))
        assert(seqs.forall(_.nested.isEmpty))
        val initArgTypes = inits.map(i => i.value.args.map(_.pType).toArray).transpose
          .map(ts => getNestedElementPTypes(ts))
        val seqArgTypes = seqs.map(i => i.value.args.map(_.pType).toArray).transpose
          .map(ts => getNestedElementPTypes(ts))
        virt.defaultSignature.toPhysical(initArgTypes, seqArgTypes).singletonContainer
    }
  }

  def getNestedElementPTypes(ptypes: Seq[PType]): PType = {
    assert(ptypes.forall(_.virtualType == ptypes.head.virtualType))
    getNestedElementPTypesOfSameType(ptypes: Seq[PType])
  }

  def getNestedElementPTypesOfSameType(ptypes: Seq[PType]): PType = {
    ptypes.head match {
      case _: PStream =>
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PStream].elementType))
        PStream(elementType, ptypes.forall(_.required))
      case _: PCanonicalArray =>
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PArray].elementType))
        PCanonicalArray(elementType, ptypes.forall(_.required))
      case _: PCanonicalSet =>
        val elementType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PSet].elementType))
        PCanonicalSet(elementType, ptypes.forall(_.required))
      case x: PCanonicalStruct =>
        PCanonicalStruct(ptypes.forall(_.required), x.fieldNames.map(fieldName =>
          fieldName -> getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PStruct].field(fieldName).typ))
        ): _*)
      case x: PCanonicalTuple =>
        PCanonicalTuple(x._types.map(pTupleField =>
          pTupleField.copy(typ = getNestedElementPTypesOfSameType(ptypes.map { case t: PTuple => t._types(t.fieldIndex(pTupleField.index)).typ }))
        ), ptypes.forall(_.required))
      case _: PCanonicalDict =>
        val keyType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PDict].keyType))
        val valueType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PDict].valueType))
        PCanonicalDict(keyType, valueType, ptypes.forall(_.required))
      case _: PCanonicalInterval =>
        val pointType = getNestedElementPTypesOfSameType(ptypes.map(_.asInstanceOf[PInterval].pointType))
        PCanonicalInterval(pointType, ptypes.forall(_.required))
      case _ => ptypes.head.setRequired(ptypes.forall(_.required))
    }
  }

  def apply(ir: IR, env: Env[PType]): Unit = apply(ir, env, null, null, null)

  private type AAB[T] = Array[ArrayBuilder[RecursiveArrayBuilderElement[T]]]

  case class RecursiveArrayBuilderElement[T](value: T, nested: Option[AAB[T]])

  def newBuilder[T](n: Int): AAB[T] = Array.fill(n)(new ArrayBuilder[RecursiveArrayBuilderElement[T]])

  def apply(ir: IR, env: Env[PType], aggs: Array[AggStatePhysicalSignature], inits: AAB[InitOp], seqs: AAB[SeqOp]): Unit = {
    try {
      _apply(ir, env, aggs, inits, seqs)
    } catch {
      case e: Exception =>
        throw new RuntimeException(s"error while inferring IR:\n${Pretty(ir)}", e)
    }
    VisitIR(ir) { case (node: IR) =>
      if (node._pType == null)
        throw new RuntimeException(s"ptype inference failure: node not inferred:\n${Pretty(node)}\n ** Full IR: **\n${Pretty(ir)}")
    }
  }
  private def _apply(ir: IR, env: Env[PType], aggs: Array[AggStatePhysicalSignature], inits: AAB[InitOp], seqs: AAB[SeqOp]): Unit = {
    if (ir._pType != null)
      throw new RuntimeException(ir.toString)

    def infer(ir: IR, env: Env[PType] = env, aggs: Array[AggStatePhysicalSignature] = aggs,
      inits: AAB[InitOp] = inits, seqs: AAB[SeqOp] = seqs): Unit = _apply(ir, env, aggs, inits, seqs)

    ir._pType = ir match {
      case I32(_) => PInt32(true)
      case I64(_) => PInt64(true)
      case F32(_) => PFloat32(true)
      case F64(_) => PFloat64(true)
      case Str(_) => PString(true)
      case Literal(t, _) => PType.canonical(t, true)
      case True() | False() => PBoolean(true)
      case Cast(ir, t) =>
        infer(ir)
        PType.canonical(t, ir.pType.required)
      case CastRename(ir, t) =>
        infer(ir)
        ir.pType.deepRename(t)
      case NA(t) =>
        PType.canonical(t).deepInnerRequired(false)
      case Die(msg, t) =>
        infer(msg)
        PType.canonical(t).deepInnerRequired(true)
      case IsNA(ir) =>
        infer(ir)
        PBoolean(true)
      case Ref(name, _) => env.lookup(name)
      case MakeNDArray(data, shape, rowMajor) =>
        infer(data)
        infer(shape)
        infer(rowMajor)

        val nElem = shape.pType.asInstanceOf[PTuple].size

        PNDArray(coerce[PArray](data.pType).elementType.setRequired(true), nElem, data.pType.required && shape.pType.required)
      case StreamRange(start: IR, stop: IR, step: IR) =>
        infer(start)
        infer(stop)
        infer(step)

        assert(start.pType isOfType stop.pType)
        assert(start.pType isOfType step.pType)

        val allRequired = start.pType.required && stop.pType.required && step.pType.required
        PStream(start.pType.setRequired(true), allRequired)
      case ArrayZeros(length) =>
        infer(length)
        PCanonicalArray(PInt32(true), length.pType.required)
      case ArrayLen(a: IR) =>
        infer(a)

        PInt32(a.pType.required)
      case LowerBoundOnOrderedCollection(orderedCollection: IR, bound: IR, _) =>
        infer(orderedCollection)
        infer(bound)

        PInt32(orderedCollection.pType.required)
      case Let(name, value, body) =>
        infer(value)
        infer(body, env.bind(name, value.pType))
        body.pType
      case TailLoop(_, args, body) =>
        args.foreach { case (_, ir) => infer(ir) }
        infer(body, env.bind(args.map { case (n, ir) => n -> ir.pType }: _*))
        body.pType
      case Recur(_, args, typ) =>
        args.foreach { a => infer(a) }
        PType.canonical(typ)
      case ApplyBinaryPrimOp(op, l, r) =>
        infer(l)
        infer(r)

        val vType = BinaryOp.getReturnType(op, l.pType.virtualType, r.pType.virtualType)
        PType.canonical(vType, l.pType.required && r.pType.required)
      case ApplyUnaryPrimOp(op, v) =>
        infer(v)
        PType.canonical(UnaryOp.getReturnType(op, v.pType.virtualType)).setRequired(v.pType.required)
      case ApplyComparisonOp(op, l, r) =>
        infer(l)
        infer(r)

        assert(l.pType isOfType r.pType)
        op match {
          case _: Compare => PInt32(l.pType.required && r.pType.required)
          case _ => PBoolean(l.pType.required && r.pType.required)
        }
      case a: AbstractApplyNode[_] =>
        val pTypes = a.args.map(i => {
          infer(i)
          i.pType
        })
        a.implementation.returnPType(pTypes, a.returnType)
      case a@ApplySpecial(_, args, _) =>
        val pTypes = args.map(i => {
          infer(i)
          i.pType
        })
        a.implementation.returnPType(pTypes, a.returnType)
      case ArrayRef(a, i, s) =>
        infer(a)
        infer(i)
        infer(s)
        assert(i.pType isOfType PInt32())

        val aType = coerce[PArray](a.pType)
        val elemType = aType.elementType
        elemType.orMissing(a.pType.required && i.pType.required)
      case ArraySort(a, leftName, rightName, compare) =>
        infer(a)
        val et = coerce[PStream](a.pType).elementType

        infer(compare, env.bind(leftName -> et, rightName -> et))
        assert(compare.pType.isOfType(PBoolean()))

        PCanonicalArray(et, a.pType.required)
      case ToSet(a) =>
        infer(a)
        val et = coerce[PIterable](a.pType).elementType
        PSet(et, a.pType.required)
      case ToDict(a) =>
        infer(a)
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType).elementType)
        // Dict key/value types don't depend on PIterable's requiredeness because we have an interface guarantee that
        // null PIterables are filtered out before dict construction
        val keyRequired = elt.types(0).required
        val valRequired = elt.types(1).required
        PDict(elt.types(0).setRequired(keyRequired), elt.types(1).setRequired(valRequired), a.pType.required)
      case ToArray(a) =>
        infer(a)
        val elt = coerce[PIterable](a.pType).elementType
        PArray(elt, a.pType.required)
      case CastToArray(a) =>
        infer(a)
        val elt = coerce[PIterable](a.pType).elementType
        PArray(elt, a.pType.required)
      case ToStream(a) =>
        infer(a)
        val elt = coerce[PIterable](a.pType).elementType
        PStream(elt, a.pType.required)
      case GroupByKey(collection) =>
        infer(collection)
        val elt = coerce[PBaseStruct](coerce[PStream](collection.pType).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection.pType.required)
      case StreamMap(a, name, body) =>
        infer(a)
        infer(body, env.bind(name, a.pType.asInstanceOf[PStream].elementType))
        PStream(body.pType, a.pType.required)
      case StreamZip(as, names, body, behavior) =>
        as.foreach(infer(_))
        val e = behavior match {
          case ArrayZipBehavior.ExtendNA =>
            env.bindIterable(names.zip(as.map(a => -a.pType.asInstanceOf[PStream].elementType)))
          case _ =>
            env.bindIterable(names.zip(as.map(a => a.pType.asInstanceOf[PStream].elementType)))
        }
        infer(body, e)
        PStream(body.pType, as.forall(_.pType.required))
      case StreamFilter(a, name, cond) =>
        infer(a)
        infer(cond, env = env.bind(name, a.pType.asInstanceOf[PStream].elementType))
        a.pType
      case StreamFlatMap(a, name, body) =>
        infer(a)
        infer(body, env.bind(name, a.pType.asInstanceOf[PStream].elementType))

        // Whether an array must return depends on a, but element requiredeness depends on body (null a elements elided)
        PStream(coerce[PIterable](body.pType).elementType, a.pType.required)
      case StreamFold(a, zero, accumName, valueName, body) =>
        infer(zero)
        infer(a)
        val accType = zero.pType.orMissing(a.pType.required)
        infer(body, env.bind(accumName -> accType, valueName -> a.pType.asInstanceOf[PStream].elementType))
        if (body.pType != accType) {
          val resPType = InferPType.getNestedElementPTypes(FastSeq(body.pType, accType))
          // the below does a two-pass inference to unify the accumulator with the body ptype.
          // this is not ideal, may cause problems in the future.
          clearPTypes(body)
          infer(body, env.bind(accumName -> resPType, valueName -> a.pType.asInstanceOf[PStream].elementType))
          resPType
        } else
          accType
      case StreamFor(a, value, body) =>
        infer(a)
        infer(body, env.bind(value -> a.pType.asInstanceOf[PStream].elementType))
        PVoid
      case x@StreamFold2(a, acc, valueName, seq, res) =>
        infer(a)
        acc.foreach { case (_, accIR) => infer(accIR) }
        var seqEnv = env.bind(acc.map { case (name, accIR) => (name, accIR.pType) }: _*)
          .bind(valueName -> a.pType.asInstanceOf[PStream].elementType)
        var anyMismatch = false
        x.accPTypes = seq.zip(acc.map(_._1)).map { case (seqIR, name) =>
          infer(seqIR, seqEnv)
          if (seqIR.pType != seqEnv.lookup(name)) {
            anyMismatch = true
            val resPType = InferPType.getNestedElementPTypes(FastSeq(seqIR.pType, seqEnv.lookup(name)))
            seqEnv = seqEnv.bind(name, resPType)
            // the below does a two-pass inference to unify the accumulator with the body ptype.
            // this is not ideal, may cause problems in the future.
            resPType
          } else seqIR.pType
        }

        acc.indices.foreach {i =>
          clearPTypes(seq(i))
          infer(seq(i), seqEnv)
        }

        infer(res, seqEnv.delete(valueName))
        res.pType.setRequired(res.pType.required && a.pType.required)
      case x@StreamScan(a, zero, accumName, valueName, body) =>
        infer(zero)

        infer(a)
        infer(body, env.bind(accumName -> zero.pType, valueName -> a.pType.asInstanceOf[PStream].elementType))
        x.accPType = if (body.pType != zero.pType) {
          val resPType = InferPType.getNestedElementPTypes(FastSeq(body.pType, zero.pType))
          // the below does a two-pass inference to unify the accumulator with the body ptype.
          // this is not ideal, may cause problems in the future.
          clearPTypes(body)
          infer(body, env.bind(accumName -> resPType, valueName -> a.pType.asInstanceOf[PStream].elementType))
          resPType
        } else zero.pType

        PStream(elementType = x.accPType)
      case StreamLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) =>
        infer(lIR)
        infer(rIR)
        val e = env.bind(lName -> lIR.pType.asInstanceOf[PStream].elementType, rName -> -rIR.pType.asInstanceOf[PStream].elementType)

        infer(compare, e)
        infer(join, e)

        PStream(join.pType, lIR.pType.required)
      case NDArrayShape(nd) =>
        infer(nd)
        val r = nd.pType.asInstanceOf[PNDArray].shape.pType
        r.setRequired(r.required && nd.pType.required)
      case NDArrayReshape(nd, shape) =>
        infer(nd)
        infer(shape)

        val shapeT = shape.pType.asInstanceOf[PTuple]
        PNDArray(coerce[PNDArray](nd.pType).elementType, shapeT.size,
          nd.pType.required && shapeT.required && shapeT.types.forall(_.required))
      case NDArrayConcat(nds, _) =>
        infer(nds)
        val ndtyp = coerce[PNDArray](coerce[PArray](nds.pType).elementType)
        ndtyp.setRequired(nds.pType.required && ndtyp.required)
      case NDArrayMap(nd, name, body) =>
        infer(nd)
        val ndPType = nd.pType.asInstanceOf[PNDArray]
        infer(body, env.bind(name -> ndPType.elementType))

        PNDArray(body.pType.setRequired(true), ndPType.nDims, nd.pType.required)
      case NDArrayMap2(l, r, lName, rName, body) =>
        infer(l)
        infer(r)

        val lPType = l.pType.asInstanceOf[PNDArray]
        val rPType = r.pType.asInstanceOf[PNDArray]

        InferPType(body, env.bind(lName -> lPType.elementType, rName -> rPType.elementType))

        PNDArray(body.pType.setRequired(true), lPType.nDims, l.pType.required && r.pType.required)
      case NDArrayReindex(nd, indexExpr) =>
        infer(nd)

        PNDArray(coerce[PNDArray](nd.pType).elementType, indexExpr.length, nd.pType.required)
      case NDArrayRef(nd, idxs) =>
        infer(nd)

        var allRequired = nd.pType.required
        val it = idxs.iterator
        while (it.hasNext) {
          val idxIR = it.next()
          infer(idxIR)
          assert(idxIR.pType.isOfType(PInt64()) || idxIR.pType.isOfType(PInt32()))
          if (allRequired && !idxIR.pType.required) {
            allRequired = false
          }
        }

        coerce[PNDArray](nd.pType).elementType.setRequired(allRequired)
      case NDArraySlice(nd, slices) =>
        infer(nd)
        infer(slices)
        val slicesPT = coerce[PTuple](slices.pType)
        val remainingDims = slicesPT.types.filter(_.isInstanceOf[PTuple])
        PNDArray(coerce[PNDArray](nd.pType).elementType, remainingDims.length,
          slicesPT.required && slicesPT.types.forall(_.required)
            && remainingDims.iterator.flatMap(t => coerce[PTuple](t).types).forall(_.required) && nd.pType.required)
      case NDArrayFilter(nd, filters) =>
        infer(nd)
        filters.foreach(infer(_))
        coerce[PNDArray](nd.pType)
      case NDArrayMatMul(l, r) =>
        infer(l)
        infer(r)
        val lTyp = coerce[PNDArray](l.pType)
        val rTyp = coerce[PNDArray](r.pType)
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
          (name, a.pType)
        }: _ *)
      case SelectFields(old, fields) =>
        infer(old)
        val tbs = coerce[PStruct](old.pType)
        tbs.select(fields.toFastIndexedSeq)._1
      case InsertFields(old, fields, fieldOrder) =>
        infer(old)
        val tbs = coerce[PStruct](old.pType)

        val s = tbs.insertFields(fields.map(f => {
          infer(f._2)
          (f._1, f._2.pType)
        }))

        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      case GetField(o, name) =>
        infer(o)
        val t = coerce[PStruct](o.pType)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      case MakeTuple(values) =>
        PCanonicalTuple(values.map { case (idx, v) =>
          infer(v)
          PTupleField(idx, v.pType)
        }.toFastIndexedSeq, true)
      case MakeArray(irs, t) =>
        if (irs.isEmpty) {
          PType.canonical(t, true).deepInnerRequired(true)
        } else {
          val elementTypes = irs.map { elt =>
            infer(elt)
            elt.pType
          }

          val inferredElementType = getNestedElementPTypes(elementTypes)
          PArray(inferredElementType, true)
        }
      case GetTupleElement(o, idx) =>
        infer(o)
        val t = coerce[PTuple](o.pType)
        val fd = t.fields(t.fieldIndex(idx)).typ
        fd.setRequired(t.required && fd.required)
      case If(cond, cnsq, altr) =>
        infer(cond)
        infer(cnsq)
        infer(altr)

        assert(cond.pType isOfType PBoolean())

        val branchType = getNestedElementPTypes(IndexedSeq(cnsq.pType, altr.pType))

        branchType.setRequired(branchType.required && cond.pType.required)
      case Coalesce(values) =>
        getNestedElementPTypes(values.map(theIR => {
          infer(theIR)
          theIR.pType
        }))
      case In(_, pType: PType) => pType
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        infer(contextsIR)
        infer(globalsIR)
        infer(bodyIR, env.bind(contextsName -> coerce[PStream](contextsIR.pType).elementType, globalsName -> globalsIR.pType))

        PCanonicalArray(bodyIR.pType, contextsIR.pType.required)
      case ReadPartition(rowIR, codecSpec, rowType) =>
        infer(rowIR)
        val child = codecSpec.buildDecoder(rowType)._1
        PStream(child, child.required)
      case ReadValue(path, spec, requestedType) =>
        infer(path)
        spec.buildDecoder(requestedType)._1
      case WriteValue(value, pathPrefix, spec) =>
        infer(value)
        infer(pathPrefix)
        PCanonicalString(pathPrefix.pType.required && value.pType.required)
      case MakeStream(irs, t) =>
        if (irs.isEmpty) {
          PType.canonical(t, true).deepInnerRequired(true)
        } else {
          PStream(getNestedElementPTypes(irs.map(theIR => {
            infer(theIR)
            theIR.pType
          })), true)
        }
      case x@InitOp(i, args, sig, op) =>
        op match {
          case Group() =>
            val nested = sig.nested.get
            val newInits = newBuilder[InitOp](nested.length)
            val IndexedSeq(initArg) = args
            infer(initArg, env, null, inits = newInits, seqs = null)
            if (inits != null)
              inits(i) += RecursiveArrayBuilderElement(x, Some(newInits))
          case AggElementsLengthCheck() =>
            val nested = sig.nested.get
            val newInits = newBuilder[InitOp](nested.length)
            val initArg = args match {
              case Seq(len, initArg) =>
                infer(len, env, null, null, null)
                initArg
              case Seq(initArg) => initArg
            }
            infer(initArg, env, null, inits = newInits, seqs = null)
            if (inits != null)
              inits(i) += RecursiveArrayBuilderElement(x, Some(newInits))
          case _ =>
            assert(sig.nested.isEmpty)
            args.foreach(infer(_, env, null, null, null))
            if (inits != null)
              inits(i) += RecursiveArrayBuilderElement(x, None)
        }
        PVoid
      case x@SeqOp(i, args, sig, op) =>
        op match {
          case Group() =>
            val nested = sig.nested.get
            val newSeqs = newBuilder[SeqOp](nested.length)
            val IndexedSeq(k, seqArg) = args
            infer(k, env, null, inits = null, seqs = null)
            infer(seqArg, env, null, inits = null, seqs = newSeqs)
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, Some(newSeqs))
          case AggElements() =>
            val nested = sig.nested.get
            val newSeqs = newBuilder[SeqOp](nested.length)
            val IndexedSeq(idx, seqArg) = args
            infer(idx, env, null, inits = null, seqs = null)
            infer(seqArg, env, null, inits = null, seqs = newSeqs)
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, Some(newSeqs))
          case AggElementsLengthCheck() =>
            val nested = sig.nested.get
            val IndexedSeq(idx) = args
            infer(idx, env, null, inits = null, seqs = null)
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, None)
          case _ =>
            assert(sig.nested.isEmpty)
            args.foreach(infer(_, env, null, null, null))
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, None)
        }
        PVoid
      case x@ResultOp(resultIdx, sigs) =>
        PCanonicalTuple(true, (resultIdx until resultIdx + sigs.length).map(i => aggs(i).resultType): _*)
      case x@RunAgg(body, result, signature) =>
        val inits = newBuilder[InitOp](signature.length)
        val seqs = newBuilder[SeqOp](signature.length)
        infer(body, env, inits = inits, seqs = seqs, aggs = null)
        val sigs = signature.indices.map { i => computePhysicalAgg(signature(i), inits(i), seqs(i)) }.toArray
        infer(result, env, aggs = sigs, inits = null, seqs = null)
        x.physicalSignatures = sigs
        result.pType
      case x@RunAggScan(array, name, init, seq, result, signature) =>
        infer(array)
        val e2 = env.bind(name, coerce[PStream](array.pType).elementType)
        val inits = newBuilder[InitOp](signature.length)
        val seqs = newBuilder[SeqOp](signature.length)
        infer(init, env = e2, inits = inits, seqs = null, aggs = null)
        infer(seq, env = e2, inits = null, seqs = seqs, aggs = null)
        val sigs = signature.indices.map { i => computePhysicalAgg(signature(i), inits(i), seqs(i)) }.toArray
        infer(result, env = e2, aggs = sigs, inits = null, seqs = null)
        x.physicalSignatures = sigs
        PStream(result.pType, array.pType.required)
      case AggStateValue(i, sig) => PCanonicalBinary(true)
      case x if x.typ == TVoid =>
        x.children.foreach(c => infer(c.asInstanceOf[IR]))
        PVoid
      case NDArrayAgg(nd, _) =>
        infer(nd)
        PType.canonical(ir.typ)
      case x if x.typ == TVoid =>
        x.children.foreach(c => infer(c.asInstanceOf[IR]))
        PVoid
    }
    if (ir.pType.virtualType != ir.typ)
      throw new RuntimeException(s"pType.virtualType: ${ir.pType.virtualType}, vType = ${ir.typ}\n  ir=$ir")
  }
}
