package is.hail.expr.ir

import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.HailContext
import is.hail.types.{RDict, RIterable, TypeWithRequiredness}

object InferPType {

  def clearPTypes(x: BaseIR): Unit = {
    x match {
      case x: IR =>
        x._pType = null
      case _ =>
    }
    x.children.foreach(clearPTypes)
  }

  def getCompatiblePType(pTypes: Seq[PType]): PType = {
    val r = TypeWithRequiredness.apply(pTypes.head.virtualType)
    pTypes.foreach(r.fromPType)
    getCompatiblePType(pTypes, r)
  }

  def getCompatiblePType(pTypes: Seq[PType], result: TypeWithRequiredness): PType = {
    assert(pTypes.tail.forall(pt => pt.virtualType == pTypes.head.virtualType))
    if (pTypes.tail.forall(pt => pt == pTypes.head))
      pTypes.head
    else result.canonicalPType(pTypes.head.virtualType)
  }

  def apply(ir: IR): Unit = apply(ir, Env.empty)

  private type AAB[T] = Array[ArrayBuilder[RecursiveArrayBuilderElement[T]]]

  case class RecursiveArrayBuilderElement[T](value: T, nested: Option[AAB[T]])

  def newBuilder[T](n: Int): AAB[T] = Array.fill(n)(new ArrayBuilder[RecursiveArrayBuilderElement[T]])

  def apply(ir: IR, env: Env[PType]): Unit = {
    try {
      val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
      val requiredness = Requiredness.apply(ir, usesAndDefs, null, env) // Value IR inference doesn't need context
      requiredness.states.m.foreach { case (ir, types) =>
        ir.t match {
          case x: StreamFold => x.accPTypes = types.map(r => r.canonicalPType(x.zero.typ)).toArray
          case x: StreamScan => x.accPTypes = types.map(r => r.canonicalPType(x.zero.typ)).toArray
          case x: StreamFold2 =>
            x.accPTypes = x.accum.zip(types).map { case ((_, arg), r) => r.canonicalPType(arg.typ) }.toArray
          case x: TailLoop =>
            x.accPTypes = x.params.zip(types).map { case ((_, arg), r) => r.canonicalPType(arg.typ) }.toArray
        }
      }
      _inferWithRequiredness(ir, env, requiredness, usesAndDefs)
    } catch {
      case e: Exception =>
        throw new RuntimeException(s"error while inferring IR:\n${Pretty(ir)}", e)
    }
    VisitIR(ir) { case (node: IR) =>
      if (node._pType == null)
        throw new RuntimeException(s"ptype inference failure: node not inferred:\n${Pretty(node)}\n ** Full IR: **\n${Pretty(ir)}")
    }
  }

  private def lookup(name: String, r: TypeWithRequiredness, defNode: IR): PType = defNode match {
    case Let(`name`, value, _) => value.pType
    case TailLoop(`name`, _, body) => r.canonicalPType(body.typ)
    case x: TailLoop => x.accPTypes(x.paramIdx(name))
    case ArraySort(a, l, r, c) => coerce[PStream](a.pType).elementType
    case StreamMap(a, `name`, _) => coerce[PStream](a.pType).elementType
    case x@StreamZip(as, _, _, _) =>
      coerce[PStream](as(x.nameIdx(name)).pType).elementType.setRequired(r.required)
    case StreamZipJoin(as, key, `name`, _, joinF) =>
      assert(r.required)
      getCompatiblePType(as.map { a =>
        PCanonicalStruct(true, key.map { k =>
          k -> coerce[PStruct](coerce[PStream](a.pType).elementType).fieldType(k)
        }: _*)
      }, r).setRequired(true)
    case x@StreamZipJoin(as, key, _, `name`, joinF) =>
      assert(r.required)
      assert(!r.asInstanceOf[RIterable].elementType.required)
      x.getOrComputeCurValsType {
        PCanonicalArray(
          getCompatiblePType(
            as.map(a => coerce[PStruct](coerce[PStream](a.pType).elementType)),
            r.asInstanceOf[RIterable].elementType).setRequired(false),
          required = true)
      }
    case StreamFilter(a, `name`, _) => coerce[PStream](a.pType).elementType
    case StreamFlatMap(a, `name`, _) => coerce[PStream](a.pType).elementType
    case StreamFor(a, `name`, _) => coerce[PStream](a.pType).elementType
    case StreamFold(a, _, _, `name`, _) => coerce[PStream](a.pType).elementType
    case x: StreamFold => x.accPType
    case StreamScan(a, _, _, `name`, _) => coerce[PStream](a.pType).elementType
    case x: StreamScan => x.accPType
    case StreamFold2(a, _, `name`, _, _) => coerce[PStream](a.pType).elementType
    case x: StreamFold2 => x.accPTypes(x.nameIdx(name))
    case StreamJoinRightDistinct(left, _, _, _, `name`, _, _, joinType) =>
      coerce[PStream](left.pType).elementType.orMissing(joinType == "left")
    case StreamJoinRightDistinct(_, right, _, _, _, `name`, _, _) =>
      coerce[PStream](right.pType).elementType.setRequired(false)
    case RunAggScan(a, `name`, _, _, _, _) => coerce[PStream](a.pType).elementType
    case NDArrayMap(nd, `name`, _) => coerce[PNDArray](nd.pType).elementType
    case NDArrayMap2(left, _, `name`, _, _) => coerce[PNDArray](left.pType  ).elementType
    case NDArrayMap2(_, right, _, `name`, _) => coerce[PNDArray](right.pType).elementType
    case x@CollectDistributedArray(_, _, `name`, _, _) => x.decodedContextPType
    case x@CollectDistributedArray(_, _, _, `name`, _) => x.decodedGlobalPType
    case x@ShuffleWith(_, _, _, _, `name`, _, _) => x.shufflePType
    case _ => throw new RuntimeException(s"$name not found in definition \n${ Pretty(defNode) }")
  }

  private def _inferWithRequiredness(node: IR, env: Env[PType], requiredness: RequirednessAnalysis, usesAndDefs: UsesAndDefs): Unit = {
    if (node._pType != null)
      throw new RuntimeException(node.toString)
    node.children.foreach {
      case x: IR => _inferWithRequiredness(x, env, requiredness, usesAndDefs)
      case c => throw new RuntimeException(s"unsupported node:\n${Pretty(c)}")
    }
    node._pType = node match {
      case x if x.typ == TVoid => PVoid
      case _: I32 | _: I64 | _: F32 | _: F64 | _: Str | _: UUID4 | _: Literal | _: True | _: False
           | _: Cast | _: NA | _: Die | _: IsNA | _: ArrayZeros | _: ArrayLen | _: StreamLen
           | _: LowerBoundOnOrderedCollection | _: ApplyBinaryPrimOp
           | _: ApplyUnaryPrimOp | _: ApplyComparisonOp | _: WriteValue
           | _: NDArrayAgg | _: ShuffleWrite | _: AggStateValue | _: CombOpValue | _: InitFromSerializedValue =>
        requiredness(node).canonicalPType(node.typ)
      case CastRename(v, typ) => v.pType.deepRename(typ)
      case x: BaseRef if usesAndDefs.free.contains(RefEquality(x)) =>
        env.lookup(x.name)
      case x: BaseRef =>
        lookup(x.name, requiredness(node), usesAndDefs.defs.lookup(node).asInstanceOf[IR])
      case MakeNDArray(data, shape, rowMajor) =>
        val nElem = shape.pType.asInstanceOf[PTuple].size
        PCanonicalNDArray(coerce[PArray](data.pType).elementType.setRequired(true), nElem, requiredness(node).required)
      case StreamRange(start: IR, stop: IR, step: IR) =>
        assert(start.pType isOfType stop.pType)
        assert(start.pType isOfType step.pType)
        PCanonicalStream(start.pType.setRequired(true), requiredness(node).required)
      case Let(_, _, body) => body.pType
      case TailLoop(_, _, body) => body.pType
      case a: AbstractApplyNode[_] => a.implementation.returnPType(a.returnType, a.args.map(_.pType))
      case ArrayRef(a, i, s) =>
        assert(i.pType isOfType PInt32())
        coerce[PArray](a.pType).elementType.setRequired(requiredness(node).required)
      case ArraySort(a, leftName, rightName, lessThan) =>
        assert(lessThan.pType.isOfType(PBoolean()))
        PCanonicalArray(coerce[PIterable](a.pType).elementType, requiredness(node).required)
      case ToSet(a) =>
        PCanonicalSet(coerce[PIterable](a.pType).elementType, requiredness(node).required)
      case ToDict(a) =>
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType).elementType)
        PCanonicalDict(elt.types(0), elt.types(1), requiredness(node).required)
      case ToArray(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PCanonicalArray(elt, requiredness(node).required)
      case CastToArray(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PCanonicalArray(elt, requiredness(node).required)
      case ToStream(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PCanonicalStream(elt, requiredness(node).required)
      case GroupByKey(collection) =>
        val r = coerce[RDict](requiredness(node))
        val elt = coerce[PBaseStruct](coerce[PStream](collection.pType).elementType)
        PCanonicalDict(elt.types(0), PCanonicalArray(elt.types(1), r.valueType.required), r.required)
      case StreamTake(a, len) =>
        a.pType.setRequired(requiredness(node).required)
      case StreamDrop(a, len) =>
        a.pType.setRequired(requiredness(node).required)
      case StreamGrouped(a, size) =>
        val r = coerce[RIterable](requiredness(node))
        assert(size.pType isOfType PInt32())
        assert(a.pType.isInstanceOf[PStream])
        PCanonicalStream(a.pType.setRequired(r.elementType.required), r.required)
      case StreamGroupByKey(a, key) =>
        val r = coerce[RIterable](requiredness(node))
        val structType = a.pType.asInstanceOf[PStream].elementType.asInstanceOf[PStruct]
        assert(structType.required)
        PCanonicalStream(a.pType.setRequired(r.elementType.required), r.required)
      case StreamMap(a, name, body) =>
        PCanonicalStream(body.pType, requiredness(node).required)
      case StreamMerge(left, right, key) =>
        val r = coerce[RIterable](requiredness(node))
        val leftEltType = coerce[PStream](left.pType).elementType
        val rightEltType = coerce[PStream](right.pType).elementType
        PCanonicalStream(getCompatiblePType(Seq(leftEltType, rightEltType), r.elementType), r.required)
      case StreamZip(as, names, body, behavior) =>
        PCanonicalStream(body.pType, requiredness(node).required)
      case StreamZipJoin(as, _, curKey, curVals, joinF) =>
        val r = requiredness(node).asInstanceOf[RIterable]
        val rEltType = joinF.pType
        val eltTypes = as.map(_.pType.asInstanceOf[PStream].elementType)
        assert(eltTypes.forall(_.required))
        PCanonicalStream(rEltType, r.required)
      case StreamMultiMerge(as, _) =>
        val r = coerce[RIterable](requiredness(node))
        val eltTypes = as.map(_.pType.asInstanceOf[PStream].elementType)
        assert(eltTypes.forall(_.required))
        assert(r.elementType.required)
        PCanonicalStream(
          getCompatiblePType(as.map(_.pType.asInstanceOf[PStream].elementType), r.elementType),
          r.required)
      case StreamFilter(a, name, cond) => a.pType
      case StreamFlatMap(a, name, body) =>
        PCanonicalStream(coerce[PIterable](body.pType).elementType, requiredness(node).required)
      case x: StreamFold =>
        x.accPType.setRequired(requiredness(node).required)
      case x: StreamFold2 =>
        x.result.pType.setRequired(requiredness(node).required)
      case x: StreamScan =>
        val r = coerce[RIterable](requiredness(node))
        PCanonicalStream(x.accPType.setRequired(r.elementType.required), r.required)
      case StreamJoinRightDistinct(_, _, _, _, _, _, join, _) =>
        PCanonicalStream(join.pType, requiredness(node).required)
      case NDArrayShape(nd) =>
        val r = nd.pType.asInstanceOf[PNDArray].shape.pType
        r.setRequired(requiredness(node).required)
      case NDArrayReshape(nd, shape) =>
        val shapeT = shape.pType.asInstanceOf[PTuple]
        PCanonicalNDArray(coerce[PNDArray](nd.pType).elementType, shapeT.size,
          requiredness(node).required)
      case NDArrayConcat(nds, _) =>
        val ndtyp = coerce[PNDArray](coerce[PArray](nds.pType).elementType)
        ndtyp.setRequired(requiredness(node).required)
      case NDArrayMap(nd, name, body) =>
        val ndPType = nd.pType.asInstanceOf[PNDArray]
        PCanonicalNDArray(body.pType.setRequired(true), ndPType.nDims, requiredness(node).required)
      case NDArrayMap2(l, r, lName, rName, body) =>
        val lPType = l.pType.asInstanceOf[PNDArray]
        PCanonicalNDArray(body.pType.setRequired(true), lPType.nDims, requiredness(node).required)
      case NDArrayReindex(nd, indexExpr) =>
        PCanonicalNDArray(coerce[PNDArray](nd.pType).elementType, indexExpr.length, requiredness(node).required)
      case NDArrayRef(nd, idxs) =>
        coerce[PNDArray](nd.pType).elementType.setRequired(requiredness(node).required)
      case NDArraySlice(nd, slices) =>
        val remainingDims = coerce[PTuple](slices.pType).types.filter(_.isInstanceOf[PTuple])
        PCanonicalNDArray(coerce[PNDArray](nd.pType).elementType, remainingDims.length, requiredness(node).required)
      case NDArrayFilter(nd, filters) => coerce[PNDArray](nd.pType)
      case NDArrayMatMul(l, r) =>
        val lTyp = coerce[PNDArray](l.pType)
        val rTyp = coerce[PNDArray](r.pType)
        PCanonicalNDArray(lTyp.elementType, TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), requiredness(node).required)
      case NDArrayQR(nd, mode) => NDArrayQR.pTypes(mode)
      case MakeStruct(fields) =>
        PCanonicalStruct(requiredness(node).required,
          fields.map { case (name, a) => (name, a.pType) }: _ *)
      case SelectFields(old, fields) =>
        if(HailContext.getFlag("use_spicy_ptypes") != null) {
          PSubsetStruct(coerce[PStruct](old.pType), fields:_*)
        } else {
          val tbs = coerce[PStruct](old.pType)
          tbs.selectFields(fields.toFastIndexedSeq)
        }
      case InsertFields(old, fields, fieldOrder) =>
        val tbs = coerce[PStruct](old.pType)
        val s = tbs.insertFields(fields.map(f => { (f._1, f._2.pType) }))
        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PCanonicalStruct(tbs.required, fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      case GetField(o, name) =>
        val t = coerce[PStruct](o.pType)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        t.field(name).typ.setRequired(requiredness(node).required)
      case MakeTuple(values) =>
        PCanonicalTuple(values.map { case (idx, v) =>
          PTupleField(idx, v.pType)
        }.toFastIndexedSeq, requiredness(node).required)
      case MakeArray(irs, t) =>
        val r = coerce[RIterable](requiredness(node))
        if (irs.isEmpty) r.canonicalPType(t) else
        PCanonicalArray(getCompatiblePType(irs.map(_.pType), r.elementType), r.required)
      case GetTupleElement(o, idx) =>
        val t = coerce[PTuple](o.pType)
        t.fields(t.fieldIndex(idx)).typ.setRequired(requiredness(node).required)
      case If(cond, cnsq, altr) =>
        assert(cond.pType isOfType PBoolean())
        val r = requiredness(node)
        getCompatiblePType(FastIndexedSeq(cnsq.pType, altr.pType), r).setRequired(r.required)
      case Coalesce(values) =>
        val r = requiredness(node)
        getCompatiblePType(values.map(_.pType), r).setRequired(r.required)
      case In(_, pType: PType) => pType
      case x: CollectDistributedArray =>
        PCanonicalArray(x.decodedBodyPType, requiredness(node).required)
      case ReadPartition(context, rowType, reader) =>
        val child = reader.rowPType(rowType)
        PCanonicalStream(child, requiredness(node).required)
      case WritePartition(value, writeCtx, writer) =>
        writer.returnPType(writeCtx.pType, coerce[PStream](value.pType))
      case ReadValue(path, spec, requestedType) =>
        spec.decodedPType(requestedType).setRequired(requiredness(node).required)
      case MakeStream(irs, t) =>
        val r = coerce[RIterable](requiredness(node))
        if (irs.isEmpty) r.canonicalPType(t) else
          PCanonicalStream(getCompatiblePType(irs.map(_.pType), r.elementType), r.required)
      case x@ResultOp(resultIdx, sigs) =>
        PCanonicalTuple(true, sigs.map(_.pResultType): _*)
      case x@RunAgg(body, result, signature) => result.pType
      case x@RunAggScan(array, name, init, seq, result, signature) =>
        PCanonicalStream(result.pType, array.pType.required)
      case ShuffleWith(keyFields, rowType, rowEType, keyEType, name, writer, readers) =>
        val r = requiredness(node)
        assert(r.required == readers.pType.required)
        readers.pType
      case ShuffleWrite(id, rows) =>
        val r = requiredness(node)
        assert(r.required)
        PCanonicalBinary(true)
      case ShufflePartitionBounds(id, nPartitions) =>
        val r = requiredness(node)
        assert(r.required)
        PCanonicalStream(
          coerce[TShuffle](id.typ).keyDecodedPType,
          true)
      case ShuffleRead(id, keyRange) =>
        val r = requiredness(node)
        assert(r.required)
        PCanonicalStream(
          coerce[TShuffle](id.typ).rowDecodedPType,
          true)
    }
    if (node.pType.virtualType != node.typ)
      throw new RuntimeException(s"pType.virtualType: ${node.pType.virtualType}, vType = ${node.typ}\n  ir=$node")
  }
}
