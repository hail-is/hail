package is.hail.expr.ir

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir.defs._
import is.hail.types._
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat._
import scala.collection.mutable

object PruneDeadFields extends Logging {

  class ComputeMutableState {
    val requestedType: Memo[BaseType] = Memo.empty

    val relationalRefs: mutable.Map[Name, mutable.Builder[Type, IndexedSeq[Type]]] =
      mutable.HashMap.empty.withDefault((_: Name) => ArraySeq.newBuilder[Type])

    def rebuildState: RebuildMutableState =
      RebuildMutableState(requestedType, mutable.HashMap.empty)
  }

  case class RebuildMutableState(
    requestedType: Memo[BaseType],
    relationalRefs: mutable.HashMap[Name, Type],
  )

  object TypeState {
    def apply(origType: Type): TypeState = new TypeState(origType)
  }

  class TypeState(val origType: Type) {
    private var _newType: Type = null
    def newType: Type = if (_newType == null) minimal(origType) else _newType
    def newStructType: TStruct = newType.asInstanceOf[TStruct]
    def isUndefined: Boolean = _newType == null

    def union(requestedType: Type): this.type = {
      _newType = if (_newType == null) requestedType else unify(origType, _newType, requestedType)
      this
    }

    def requireFields(fields: IndexedSeq[String]): TypeState =
      union(selectKey(origType.asInstanceOf[TStruct], fields))

    def requireFieldsInElt(fields: IndexedSeq[String]): TypeState = origType match {
      case TArray(eltType) => union(TArray(selectKey(eltType.asInstanceOf[TStruct], fields)))
      case TStream(eltType) => union(TStream(selectKey(eltType.asInstanceOf[TStruct], fields)))
    }

  }

  def subsetType(t: Type, path: Array[String], index: Int = 0): Type = {
    if (index == path.length)
      PruneDeadFields.minimal(t)
    else
      t match {
        case ts: TStruct =>
          TStruct(path(index) -> subsetType(ts.field(path(index)).typ, path, index + 1))
        case ta: TArray => TArray(subsetType(ta.elementType, path, index))
        case ts: TStream => TStream(subsetType(ts.elementType, path, index))
      }
  }

  def isSupertype(superType: BaseType, subType: BaseType): Boolean = {
    try {
      (superType, subType) match {
        case (tt1: TableType, tt2: TableType) =>
          isSupertype(tt1.globalType, tt2.globalType) &&
          isSupertype(tt1.rowType, tt2.rowType) &&
          tt2.key.startsWith(tt1.key)
        case (mt1: MatrixType, mt2: MatrixType) =>
          isSupertype(mt1.globalType, mt2.globalType) &&
          isSupertype(mt1.rowType, mt2.rowType) &&
          isSupertype(mt1.colType, mt2.colType) &&
          isSupertype(mt1.entryType, mt2.entryType) &&
          mt2.rowKey.startsWith(mt1.rowKey) &&
          mt2.colKey.startsWith(mt1.colKey)
        case (TArray(et1), TArray(et2)) => isSupertype(et1, et2)
        case (TNDArray(et1, ndims1), TNDArray(et2, ndims2)) =>
          (ndims1 == ndims2) && isSupertype(et1, et2)
        case (TStream(et1), TStream(et2)) => isSupertype(et1, et2)
        case (TSet(et1), TSet(et2)) => isSupertype(et1, et2)
        case (TDict(kt1, vt1), TDict(kt2, vt2)) => isSupertype(kt1, kt2) && isSupertype(vt1, vt2)
        case (s1: TStruct, s2: TStruct) =>
          var idx = -1
          s1.fields.forall { f =>
            val s2field = s2.field(f.name)
            if (s2field.index > idx) {
              idx = s2field.index
              isSupertype(f.typ, s2field.typ)
            } else
              false
          }
        case (t1: TTuple, t2: TTuple) =>
          var idx = -1
          t1._types.forall { f =>
            val t2field = t2.fields(t2.fieldIndex(f.index))
            if (t2field.index > idx) {
              idx = t2field.index
              isSupertype(f.typ, t2field.typ)
            } else {
              false
            }
          }
        case (t1: Type, t2: Type) => t1 == t2
        case _ => fatal(s"invalid comparison: $superType / $subType")
      }
    } catch {
      case e: Throwable =>
        fatal(s"error while checking subtype:\n  super: $superType\n  sub:   $subType", e)
    }
  }

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time {
      try {
        val irCopy = ir.deepCopy()
        val ms = new ComputeMutableState
        irCopy match {
          case mir: MatrixIR =>
            memoizeMatrixIR(ctx, mir, mir.typ, ms)
            rebuild(ctx, mir, ms.rebuildState)
          case tir: TableIR =>
            memoizeTableIR(ctx, tir, tir.typ, ms)
            rebuild(ctx, tir, ms.rebuildState)
          case bmir: BlockMatrixIR =>
            memoizeBlockMatrixIR(ctx, bmir, bmir.typ, ms)
            rebuild(ctx, bmir, ms.rebuildState)
          case vir: IR =>
            memoizeValueIR(ctx, vir, vir.typ, ms)
            rebuildIR(
              ctx,
              vir,
              BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)),
              ms.rebuildState,
            )
        }
      } catch {
        case e: Throwable =>
          fatal(s"error trying to rebuild IR:\n${Pretty(ctx, ir, allowUnboundRefs = true)}", e)
      }
    }

  def selectKey(t: TStruct, k: IndexedSeq[String]): TStruct = t.filterSet(k.toSet)._1

  private def minimal(tt: TableType): TableType =
    TableType(
      rowType = TStruct.empty,
      key = FastSeq(),
      globalType = TStruct.empty,
    )

  private def minimal(mt: MatrixType): MatrixType = {
    MatrixType(
      rowKey = FastSeq(),
      colKey = FastSeq(),
      rowType = TStruct.empty,
      colType = TStruct.empty,
      globalType = TStruct.empty,
      entryType = TStruct.empty,
    )
  }

  private def minimal[T <: Type](base: T): T = {
    val result = base match {
      case _: TStruct => TStruct.empty
      case ta: TArray => TArray(minimal(ta.elementType))
      case ta: TStream => TStream(minimal(ta.elementType))
      case t => t
    }
    result.asInstanceOf[T]
  }

  private def minimalBT[T <: BaseType](base: T): T =
    (base match {
      case tt: TableType => minimal(tt)
      case mt: MatrixType => minimal(mt)
      case t: Type => minimal(t)
    }).asInstanceOf[T]

  private def unifyKey(children: Seq[IndexedSeq[String]]): IndexedSeq[String] =
    children.foldLeft(FastSeq[String]()) { case (comb, k) =>
      if (k.length > comb.length) k else comb
    }

  private def unifyBaseTypeSeq(base: BaseType, _children: Seq[BaseType]): BaseType = {
    try {
      if (_children.isEmpty)
        return minimalBT(base)
      val children = _children.toArray
      base match {
        case tt: TableType =>
          val ttChildren = children.map(_.asInstanceOf[TableType])
          tt.copy(
            key = unifyKey(ttChildren.map(_.key)),
            rowType = unify(tt.rowType, ttChildren.map(_.rowType): _*),
            globalType = unify(tt.globalType, ttChildren.map(_.globalType): _*),
          )
        case mt: MatrixType =>
          val mtChildren = children.map(_.asInstanceOf[MatrixType])
          mt.copy(
            rowKey = unifyKey(mtChildren.map(_.rowKey)),
            colKey = unifyKey(mtChildren.map(_.colKey)),
            globalType = unifySeq(mt.globalType, mtChildren.map(_.globalType)),
            rowType = unifySeq(mt.rowType, mtChildren.map(_.rowType)),
            entryType = unifySeq(mt.entryType, mtChildren.map(_.entryType)),
            colType = unifySeq(mt.colType, mtChildren.map(_.colType)),
          )
        case t: Type =>
          if (children.isEmpty)
            return minimal(t)
          t match {
            case ts: TStruct =>
              val subStructs = children.map(_.asInstanceOf[TStruct])
              val fieldArrays = ArraySeq.fill(ts.fields.length)(ArraySeq.newBuilder[Type])

              var nPresent = 0
              ts.fields.foreach { f =>
                val idx = f.index

                var found = false
                subStructs.foreach { s =>
                  s.fieldIdx.get(f.name).foreach { sIdx =>
                    if (!found) {
                      nPresent += 1
                      found = true
                    }

                    fieldArrays(idx) += s.types(sIdx)
                  }
                }
              }

              val subFields = new Array[Field](nPresent)

              var newIdx = 0
              var oldIdx = 0
              while (oldIdx < fieldArrays.length) {
                val fields = fieldArrays(oldIdx).result()
                if (fields.nonEmpty) {
                  val oldField = ts.fields(oldIdx)
                  subFields(newIdx) =
                    Field(oldField.name, unifySeq(oldField.typ, fields), newIdx)
                  newIdx += 1
                }
                oldIdx += 1
              }
              TStruct(subFields)
            case tt: TTuple =>
              val subTuples = children.map(_.asInstanceOf[TTuple])

              val fieldArrays = ArraySeq.fill(tt.size)(ArraySeq.newBuilder[Type])

              var nPresent = 0

              var typIndex = 0
              while (typIndex < tt.size) {
                val tupleField = tt._types(typIndex)

                var found = false
                subTuples.foreach { s =>
                  s.fieldIndex.get(tupleField.index).foreach { sIdx =>
                    if (!found) {
                      nPresent += 1
                      found = true
                    }
                    fieldArrays(typIndex) += s.types(sIdx)
                  }
                }
                typIndex += 1
              }

              val subFields = new Array[TupleField](nPresent)

              var newIdx = 0
              var oldIdx = 0
              while (oldIdx < fieldArrays.length) {
                val fields = fieldArrays(oldIdx).result()
                if (fields.nonEmpty) {
                  val oldField = tt._types(oldIdx)
                  subFields(newIdx) =
                    TupleField(oldField.index, unifySeq(oldField.typ, fields))
                  newIdx += 1
                }
                oldIdx += 1
              }
              TTuple(subFields)
            case ta: TArray =>
              TArray(unifySeq(ta.elementType, children.map(TIterable.elementType)))
            case ts: TStream =>
              TStream(unifySeq(ts.elementType, children.map(TIterable.elementType)))
            case _ =>
              if (!children.forall(_.asInstanceOf[Type] == t)) {
                val badChildren = children.filter(c => c.asInstanceOf[Type] != t)
                  .map(c => "\n  child: " + c.asInstanceOf[Type].parsableString())
                throw new RuntimeException(
                  s"invalid unification:\n  base:  ${t.parsableString()}${badChildren.mkString("\n")}"
                )
              }
              base
          }
      }
    } catch {
      case e: RuntimeException =>
        throw new RuntimeException(
          s"failed to unify children while unifying:\n  base:  $base\n${_children.mkString("\n")}",
          e,
        )
    }
  }

  def unify[T <: BaseType](base: T, children: T*): T =
    unifyBaseTypeSeq(base, children).asInstanceOf[T]

  private def unifySeq[T <: BaseType](base: T, children: Seq[T]): T =
    unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def relationalTypeToEnv(bt: BaseType): BindingEnv[Type] = {
    val e = bt match {
      case tt: TableType =>
        Env.empty[Type]
          .bind(TableIR.rowName, tt.rowType)
          .bind(TableIR.globalName, tt.globalType)
      case mt: MatrixType =>
        Env.empty[Type]
          .bind(MatrixIR.globalName, mt.globalType)
          .bind(MatrixIR.colName, mt.colType)
          .bind(MatrixIR.rowName, mt.rowType)
          .bind(MatrixIR.entryName, mt.entryType)
    }
    BindingEnv(e, Some(e), Some(e))
  }

  def memoizeTableIR(
    ctx: ExecuteContext,
    tir: TableIR,
    requestedType: TableType,
    memo: ComputeMutableState,
  ): Unit = {
    memo.requestedType.bind(tir, requestedType)
    tir match {
      case _: TableRead =>
      case _: TableLiteral =>
      case tir: TableParallelize =>
        val typ =
          TStruct("rows" -> TArray(requestedType.rowType), "global" -> requestedType.globalType)
        createTypeStatesAndMemoize(ctx, tir, 0, typ, memo): Unit
      case _: TableRange =>
      case TableRepartition(child, _, _) => memoizeTableIR(ctx, child, requestedType, memo)
      case TableHead(child, _) => memoizeTableIR(
          ctx,
          child,
          TableType(
            key = child.typ.key,
            rowType = unify(
              child.typ.rowType,
              selectKey(child.typ.rowType, child.typ.key),
              requestedType.rowType,
            ),
            globalType = requestedType.globalType,
          ),
          memo,
        )
      case TableTail(child, _) => memoizeTableIR(
          ctx,
          child,
          TableType(
            key = child.typ.key,
            rowType = unify(
              child.typ.rowType,
              selectKey(child.typ.rowType, child.typ.key),
              requestedType.rowType,
            ),
            globalType = requestedType.globalType,
          ),
          memo,
        )

      case tir: TableGen =>
        val Seq(contextState, globalState) =
          createTypeStatesAndMemoize(ctx, tir, 2, TStream(requestedType.rowType), memo)
        // Contexts are only used in the body so we only need to keep the fields used therein
        val contextsElemType = contextState.newType
        // Globals are exported and used in body, so keep the union of the used fields
        val globalsType = globalState.union(requestedType.globalType).newType

        createTypeStatesAndMemoize(ctx, tir, 0, TStream(contextsElemType), memo): Unit
        createTypeStatesAndMemoize(ctx, tir, 1, globalsType, memo): Unit

      case TableJoin(left, right, _, joinKey) =>
        val lk =
          unifyKey(FastSeq(requestedType.key.take(left.typ.key.length), left.typ.key.take(joinKey)))
        val lkSet = lk.toSet
        val leftDep = TableType(
          key = lk,
          rowType = TStruct(left.typ.rowType.fieldNames.flatMap(f =>
            if (lkSet.contains(f))
              Some(f -> left.typ.rowType.field(f).typ)
            else
              requestedType.rowType.selfField(f).map(reqF => f -> reqF.typ)
          ): _*),
          globalType = TStruct(left.typ.globalType.fieldNames.flatMap(f =>
            requestedType.globalType.selfField(f).map(reqF => f -> reqF.typ)
          ): _*),
        )
        memoizeTableIR(ctx, left, leftDep, memo)

        val rk =
          right.typ.key.take(joinKey + math.max(0, requestedType.key.length - left.typ.key.length))
        val rightKeyFields = rk.toSet
        val rightDep = TableType(
          key = rk,
          rowType = TStruct(right.typ.rowType.fieldNames.flatMap(f =>
            if (rightKeyFields.contains(f))
              Some(f -> right.typ.rowType.field(f).typ)
            else
              requestedType.rowType.selfField(f).map(reqF => f -> reqF.typ)
          ): _*),
          globalType = TStruct(right.typ.globalType.fieldNames.flatMap(f =>
            requestedType.globalType.selfField(f).map(reqF => f -> reqF.typ)
          ): _*),
        )
        memoizeTableIR(ctx, right, rightDep, memo)
      case TableLeftJoinRightDistinct(left, right, root) =>
        val fieldDep = requestedType.rowType.selfField(root).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val rightDep = TableType(
              key = right.typ.key,
              rowType = unify(
                right.typ.rowType,
                FastSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                  FastSeq(struct): _*
              ),
              globalType = minimal(right.typ.globalType),
            )
            memoizeTableIR(ctx, right, rightDep, memo)

            val lk = unifyKey(FastSeq(left.typ.key.take(right.typ.key.length), requestedType.key))
            val leftDep = TableType(
              key = lk,
              rowType = unify(
                left.typ.rowType,
                requestedType.rowType.filterSet(Set(root), include = false)._1,
                selectKey(left.typ.rowType, lk),
              ),
              globalType = requestedType.globalType,
            )
            memoizeTableIR(ctx, left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(ctx, left, requestedType, memo)
        }
      case TableIntervalJoin(left, right, root, product) =>
        val fieldDep = requestedType.rowType.selfField(root).map { field =>
          if (product)
            field.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
          else
            field.typ.asInstanceOf[TStruct]
        }
        fieldDep match {
          case Some(struct) =>
            val rightDep = TableType(
              key = right.typ.key,
              rowType = unify(
                right.typ.rowType,
                FastSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                  FastSeq(struct): _*
              ),
              globalType = minimal(right.typ.globalType),
            )
            memoizeTableIR(ctx, right, rightDep, memo)

            val lk = unifyKey(FastSeq(left.typ.key.take(right.typ.key.length), requestedType.key))
            val leftDep = TableType(
              key = lk,
              rowType = unify(
                left.typ.rowType,
                requestedType.rowType.filterSet(Set(root), include = false)._1,
                selectKey(left.typ.rowType, lk),
              ),
              globalType = requestedType.globalType,
            )
            memoizeTableIR(ctx, left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(ctx, left, requestedType, memo)
        }
      case TableMultiWayZipJoin(children, fieldName, globalName) =>
        val gType = requestedType.globalType.selfField(globalName)
          .map(_.typ.asInstanceOf[TArray].elementType)
          .getOrElse(TStruct.empty).asInstanceOf[TStruct]
        val rType = requestedType.rowType.selfField(fieldName)
          .map(_.typ.asInstanceOf[TArray].elementType)
          .getOrElse(TStruct.empty).asInstanceOf[TStruct]
        val child1 = children.head
        val dep = TableType(
          key = child1.typ.key,
          rowType = TStruct(child1.typ.rowType.fieldNames.flatMap(f =>
            child1.typ.keyType.selfField(f).orElse(rType.selfField(f)).map(reqF => f -> reqF.typ)
          ): _*),
          globalType = gType,
        )
        children.foreach(memoizeTableIR(ctx, _, dep, memo))
      case TableExplode(child, path) =>
        def getExplodedField(typ: TableType): Type = typ.rowType.queryTyped(path.toList)._1

        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExlosionFieldType =
          try {
            val t = getExplodedField(requestedType)
            preExplosionFieldType match {
              case _: TArray => TArray(t)
              case ts: TSet => ts.copy(elementType = t)
            }
          } catch {
            case _: AnnotationPathException => minimal(preExplosionFieldType)
          }
        val dep = requestedType.copy(rowType =
          unify(
            child.typ.rowType,
            requestedType.rowType.insert(prunedPreExlosionFieldType, path)._1.asInstanceOf[TStruct],
          )
        )
        memoizeTableIR(ctx, child, dep, memo)
      case TableFilter(child, pred) =>
        val irDep = memoizeAndGetDep(ctx, tir, 1, pred.typ, memo)
        memoizeTableIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case TableKeyBy(child, _, isSorted) =>
        val reqKey = requestedType.key
        val isPrefix = reqKey.zip(child.typ.key).forall { case (l, r) => l == r }
        val childReqKey = if (isSorted)
          child.typ.key
        else if (isPrefix)
          if (reqKey.length <= child.typ.key.length) reqKey else child.typ.key
        else FastSeq()

        memoizeTableIR(
          ctx,
          child,
          TableType(
            key = childReqKey,
            rowType = unify(
              child.typ.rowType,
              selectKey(child.typ.rowType, childReqKey),
              requestedType.rowType,
            ),
            globalType = requestedType.globalType,
          ),
          memo,
        )
      case TableOrderBy(child, sortFields) =>
        val k =
          if (
            sortFields.forall(_.sortOrder == Ascending) && child.typ.key.startsWith(
              sortFields.map(_.field)
            )
          )
            child.typ.key
          else
            FastSeq()
        memoizeTableIR(
          ctx,
          child,
          TableType(
            key = k,
            rowType = unify(
              child.typ.rowType,
              selectKey(child.typ.rowType, sortFields.map(_.field) ++ k),
              requestedType.rowType,
            ),
            globalType = requestedType.globalType,
          ),
          memo,
        )
      case TableDistinct(child) =>
        val dep = TableType(
          key = child.typ.key,
          rowType = unify(
            child.typ.rowType,
            requestedType.rowType,
            selectKey(child.typ.rowType, child.typ.key),
          ),
          globalType = requestedType.globalType,
        )
        memoizeTableIR(ctx, child, dep, memo)
      case TableMapPartitions(child, _, _, body, requestedKey, _) =>
        val requestedKeyStruct =
          child.typ.keyType.truncate(math.max(requestedType.key.length, requestedKey))
        val reqRowsType =
          unify(body.typ, TStream(requestedType.rowType), TStream(requestedKeyStruct))
        val Seq(globalState, partitionState) =
          createTypeStatesAndMemoize(ctx, tir, 1, reqRowsType, memo)
        val newGlobalType = globalState.union(requestedType.globalType).newStructType
        val newRowType = elementType(partitionState.union(TStream(requestedKeyStruct)).newType)
        val dep = TableType(
          key = requestedKeyStruct.fieldNames,
          rowType = newRowType.asInstanceOf[TStruct],
          globalType = newGlobalType,
        )
        memoizeTableIR(ctx, child, dep, memo)
      case TableMapRows(child, newRow) =>
        val (reqKey, reqRowType) = if (ContainsScan(newRow)) {
          val reqRowType = unify(
            newRow.typ,
            requestedType.rowType,
            selectKey(newRow.typ.asInstanceOf[TStruct], child.typ.key),
          )
          (child.typ.key, reqRowType)
        } else {
          (requestedType.key, requestedType.rowType)
        }
        val rowDep = memoizeAndGetDep(ctx, tir, 1, reqRowType, memo)
        val dep = TableType(
          key = reqKey,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, reqKey), rowDep.rowType),
          globalType = unify(child.typ.globalType, requestedType.globalType, rowDep.globalType),
        )
        memoizeTableIR(ctx, child, dep, memo)
      case TableMapGlobals(child, _) =>
        val Seq(globalState) =
          createTypeStatesAndMemoize(ctx, tir, 1, requestedType.globalType, memo)
        memoizeTableIR(
          ctx,
          child,
          requestedType.copy(
            globalType = globalState.newStructType
          ),
          memo,
        )
      case TableAggregateByKey(child, expr) =>
        val exprRequestedType =
          requestedType.rowType.filter(f => expr.typ.asInstanceOf[TStruct].hasField(f.name))._1
        val aggDep = memoizeAndGetDep(ctx, tir, 1, exprRequestedType, memo)
        memoizeTableIR(
          ctx,
          child,
          TableType(
            key = child.typ.key,
            rowType =
              unify(child.typ.rowType, aggDep.rowType, selectKey(child.typ.rowType, child.typ.key)),
            globalType = unify(child.typ.globalType, aggDep.globalType, requestedType.globalType),
          ),
          memo,
        )
      case TableKeyByAndAggregate(child, _, newKey, _, _) =>
        val keyDep = memoizeAndGetDep(ctx, tir, 2, newKey.typ, memo)
        val exprDep = memoizeAndGetDep(ctx, tir, 1, requestedType.valueType, memo)
        memoizeTableIR(
          ctx,
          child,
          TableType(
            key = FastSeq(), // note: this can deoptimize if prune runs before Simplify
            rowType = unify(child.typ.rowType, keyDep.rowType, exprDep.rowType),
            globalType = unify(
              child.typ.globalType,
              keyDep.globalType,
              exprDep.globalType,
              requestedType.globalType,
            ),
          ),
          memo,
        )
      case MatrixColsTable(child) =>
        val mtDep = minimal(child.typ).copy(
          globalType = requestedType.globalType,
          entryType = TStruct.empty,
          colType = requestedType.rowType,
          colKey = requestedType.key,
        )
        memoizeMatrixIR(ctx, child, mtDep, memo)
      case MatrixRowsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = requestedType.globalType,
          rowType = unify(
            child.typ.rowType,
            selectKey(child.typ.rowType, requestedType.key),
            requestedType.rowType,
          ),
          rowKey = requestedType.key,
        )
        memoizeMatrixIR(ctx, child, mtDep, memo)
      case MatrixEntriesTable(child) =>
        val mtDep = MatrixType(
          rowKey = requestedType.key.take(child.typ.rowKey.length),
          colKey = requestedType.key.drop(child.typ.rowKey.length),
          globalType = requestedType.globalType,
          colType = TStruct(
            child.typ.colType.fields.flatMap(f =>
              requestedType.rowType.selfField(f.name).map(f2 => f.name -> f2.typ)
            ): _*
          ),
          rowType = TStruct(
            child.typ.rowType.fields.flatMap(f =>
              requestedType.rowType.selfField(f.name).map(f2 => f.name -> f2.typ)
            ): _*
          ),
          entryType = TStruct(
            child.typ.entryType.fields.flatMap(f =>
              requestedType.rowType.selfField(f.name).map(f2 => f.name -> f2.typ)
            ): _*
          ),
        )
        memoizeMatrixIR(ctx, child, mtDep, memo)
      case TableUnion(children) =>
        memoizeTableIR(ctx, children(0), requestedType, memo)
        val noGlobals = requestedType.copy(globalType = TStruct())
        children.iterator.drop(1).foreach(memoizeTableIR(ctx, _, noGlobals, memo))
      case CastMatrixToTable(child, entriesFieldName, colsFieldName) =>
        val childDep = MatrixType(
          rowKey = requestedType.key,
          colKey = FastSeq(),
          globalType = if (requestedType.globalType.hasField(colsFieldName))
            requestedType.globalType.deleteKey(colsFieldName)
          else
            requestedType.globalType,
          colType = if (requestedType.globalType.hasField(colsFieldName))
            requestedType.globalType.field(colsFieldName).typ.asInstanceOf[
              TArray
            ].elementType.asInstanceOf[TStruct]
          else
            TStruct.empty,
          entryType = if (requestedType.rowType.hasField(entriesFieldName))
            requestedType.rowType.field(entriesFieldName).typ.asInstanceOf[
              TArray
            ].elementType.asInstanceOf[TStruct]
          else
            TStruct.empty,
          rowType = if (requestedType.rowType.hasField(entriesFieldName))
            requestedType.rowType.deleteKey(entriesFieldName)
          else
            requestedType.rowType,
        )
        memoizeMatrixIR(ctx, child, childDep, memo)
      case TableRename(child, rowMap, globalMap) =>
        val rowMapRev = rowMap.map { case (k, v) => (v, k) }
        val globalMapRev = globalMap.map { case (k, v) => (v, k) }
        val childDep = TableType(
          rowType = requestedType.rowType.rename(rowMapRev),
          globalType = requestedType.globalType.rename(globalMapRev),
          key = requestedType.key.map(k => rowMapRev.getOrElse(k, k)),
        )
        memoizeTableIR(ctx, child, childDep, memo)
      case TableFilterIntervals(child, _, _) =>
        memoizeTableIR(
          ctx,
          child,
          requestedType.copy(
            key = child.typ.key,
            rowType = PruneDeadFields.unify(
              child.typ.rowType,
              requestedType.rowType,
              PruneDeadFields.selectKey(child.typ.rowType, child.typ.key),
            ),
          ),
          memo,
        )
      case TableToTableApply(child, _) => memoizeTableIR(ctx, child, child.typ, memo)
      case MatrixToTableApply(child, _) => memoizeMatrixIR(ctx, child, child.typ, memo)
      case BlockMatrixToTableApply(bm, aux, _) =>
        memoizeBlockMatrixIR(ctx, bm, bm.typ, memo)
        memoizeValueIR(ctx, aux, aux.typ, memo)
      case BlockMatrixToTable(child) => memoizeBlockMatrixIR(ctx, child, child.typ, memo)
      case RelationalLetTable(name, value, body) =>
        memoizeTableIR(ctx, body, requestedType, memo)
        val usages = memo.relationalRefs(name).result()
        memoizeValueIR(ctx, value, unifySeq(value.typ, usages), memo)
    }
  }

  def memoizeMatrixIR(
    ctx: ExecuteContext,
    mir: MatrixIR,
    requestedType: MatrixType,
    memo: ComputeMutableState,
  ): Unit = {
    memo.requestedType.bind(mir, requestedType)
    mir match {
      case MatrixFilterCols(child, pred) =>
        val irDep = memoizeValueIRColBindings(ctx, mir, 1, pred.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterRows(child, pred) =>
        val irDep = memoizeValueIRRowBindings(ctx, mir, 1, pred.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterEntries(child, pred) =>
        val irDep = memoizeValueIREntryBindings(ctx, mir, 1, pred.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case MatrixUnionCols(left, right, _) =>
        val leftRequestedType = requestedType.copy(
          rowKey = left.typ.rowKey,
          rowType = unify(
            left.typ.rowType,
            requestedType.rowType,
            selectKey(left.typ.rowType, left.typ.rowKey),
          ),
        )
        val rightRequestedType = requestedType.copy(
          globalType = TStruct.empty,
          rowKey = right.typ.rowKey,
          rowType = unify(
            right.typ.rowType,
            requestedType.rowType,
            selectKey(right.typ.rowType, right.typ.rowKey),
          ),
        )
        memoizeMatrixIR(ctx, left, leftRequestedType, memo)
        memoizeMatrixIR(ctx, right, rightRequestedType, memo)
      case MatrixMapEntries(child, _) =>
        val irDep = memoizeValueIREntryBindings(ctx, mir, 1, requestedType.entryType, memo)
        val depMod = requestedType.copy(entryType = TStruct.empty)
        memoizeMatrixIR(ctx, child, unify(child.typ, depMod, irDep), memo)
      case MatrixKeyRowsBy(child, _, isSorted) =>
        val reqKey = requestedType.rowKey
        val isPrefix = reqKey.zip(child.typ.rowKey).forall { case (l, r) => l == r }
        val childReqKey = if (isSorted)
          child.typ.rowKey
        else if (isPrefix)
          if (reqKey.length <= child.typ.rowKey.length) reqKey else child.typ.rowKey
        else FastSeq()

        memoizeMatrixIR(
          ctx,
          child,
          requestedType.copy(
            rowKey = childReqKey,
            rowType = unify(
              child.typ.rowType,
              requestedType.rowType,
              selectKey(child.typ.rowType, childReqKey),
            ),
          ),
          memo,
        )
      case MatrixMapRows(child, newRow) =>
        val (reqKey, reqRowType) = if (ContainsScan(newRow))
          (
            child.typ.rowKey,
            unify(
              newRow.typ,
              requestedType.rowType,
              selectKey(newRow.typ.asInstanceOf[TStruct], child.typ.rowKey),
            ),
          )
        else
          (requestedType.rowKey, requestedType.rowType)

        val irDep = memoizeValueIREntryBindings(ctx, mir, 1, reqRowType, memo)
        val depMod =
          requestedType.copy(rowType = selectKey(child.typ.rowType, reqKey), rowKey = reqKey)
        memoizeMatrixIR(ctx, child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapCols(child, _, newKey) =>
        val irDep = memoizeValueIREntryBindings(ctx, mir, 1, requestedType.colType, memo)
        val reqKey = newKey match {
          case Some(_) => FastSeq()
          case None => requestedType.colKey
        }
        val depMod =
          requestedType.copy(colType = selectKey(child.typ.colType, reqKey), colKey = reqKey)
        memoizeMatrixIR(ctx, child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapGlobals(child, _) =>
        val Seq(globalState) =
          createTypeStatesAndMemoize(ctx, mir, 1, requestedType.globalType, memo)
        memoizeMatrixIR(
          ctx,
          child,
          unify(child.typ, requestedType.copy(globalType = globalState.newStructType)),
          memo,
        )
      case MatrixRead(_, _, _, _) =>
      case MatrixLiteral(_, _) =>
      case MatrixChooseCols(child, _) =>
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType), memo)
      case MatrixCollectColsByKey(child) =>
        val colKeySet = child.typ.colKey.toSet
        val requestedColType = requestedType.colType
        val explodedDep = requestedType.copy(
          colKey = child.typ.colKey,
          colType = TStruct(child.typ.colType.fields.flatMap { f =>
            if (colKeySet.contains(f.name))
              Some(f.name -> f.typ)
            else {
              requestedColType.selfField(f.name)
                .map(requestedField =>
                  f.name -> requestedField.typ.asInstanceOf[TArray].elementType
                )
            }
          }: _*),
          rowType = requestedType.rowType,
          entryType = TStruct(requestedType.entryType.fields.map(f =>
            f.copy(typ = f.typ.asInstanceOf[TArray].elementType)
          )),
        )
        memoizeMatrixIR(ctx, child, explodedDep, memo)
      case MatrixAggregateRowsByKey(child, _, _) =>
        val Seq(entryGlobalState, entryColState, entryRowState, entryEntryState) =
          createTypeStatesAndMemoize(ctx, mir, 1, requestedType.entryType, memo)
        val Seq(rowGlobalState, rowRowState) =
          createTypeStatesAndMemoize(ctx, mir, 2, requestedType.rowValueStruct, memo)
        val childDep = MatrixType(
          rowKey = child.typ.rowKey,
          colKey = requestedType.colKey,
          entryType = entryEntryState.newStructType,
          rowType = rowRowState
            .union(entryRowState.newType)
            .requireFields(child.typ.rowKey)
            .newStructType,
          colType = entryColState.union(requestedType.colType).newStructType,
          globalType = rowGlobalState
            .union(entryGlobalState.newType)
            .union(requestedType.globalType)
            .newStructType,
        )
        memoizeMatrixIR(ctx, child, childDep, memo)
      case MatrixAggregateColsByKey(child, _, _) =>
        val Seq(entryGlobalState, entryColState, entryRowState, entryEntryState) =
          createTypeStatesAndMemoize(ctx, mir, 1, requestedType.entryType, memo)
        val Seq(colGlobalState, colColState) =
          createTypeStatesAndMemoize(ctx, mir, 2, requestedType.colValueStruct, memo)
        val childDep: MatrixType = MatrixType(
          rowKey = requestedType.rowKey,
          colKey = child.typ.colKey,
          colType = entryColState
            .union(colColState.newType)
            .requireFields(child.typ.colKey)
            .newStructType,
          globalType = entryGlobalState
            .union(colGlobalState.newType)
            .union(requestedType.globalType)
            .newStructType,
          rowType = entryRowState.union(requestedType.rowType).newStructType,
          entryType = entryEntryState.newStructType,
        )
        memoizeMatrixIR(ctx, child, childDep, memo)
      case MatrixAnnotateRowsTable(child, table, root, product) =>
        val fieldDep = requestedType.rowType.selfField(root).map { field =>
          if (product)
            field.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
          else
            field.typ.asInstanceOf[TStruct]
        }
        fieldDep match {
          case Some(struct) =>
            val tk = table.typ.key
            val tableDep = TableType(
              key = tk,
              rowType = unify(table.typ.rowType, struct, selectKey(table.typ.rowType, tk)),
              globalType = minimal(table.typ.globalType),
            )
            memoizeTableIR(ctx, table, tableDep, memo)

            val mk = unifyKey(FastSeq(child.typ.rowKey.take(tk.length), requestedType.rowKey))
            val matDep = requestedType.copy(
              rowKey = mk,
              rowType =
                unify(
                  child.typ.rowType,
                  selectKey(child.typ.rowType, mk),
                  requestedType.rowType.filterSet(Set(root), include = false)._1,
                ),
            )
            memoizeMatrixIR(ctx, child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(ctx, child, requestedType, memo)
        }
      case MatrixAnnotateColsTable(child, table, uid) =>
        val fieldDep = requestedType.colType.selfField(uid).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val tk = table.typ.key
            val tableDep = TableType(
              key = tk,
              rowType = unify(table.typ.rowType, struct, selectKey(table.typ.rowType, tk)),
              globalType = minimal(table.typ.globalType),
            )
            memoizeTableIR(ctx, table, tableDep, memo)

            val mk =
              unifyKey(FastSeq(child.typ.colKey.take(table.typ.key.length), requestedType.colKey))
            val matDep = requestedType.copy(
              colKey = mk,
              colType = unify(
                child.typ.colType,
                requestedType.colType.filterSet(Set(uid), include = false)._1,
                selectKey(child.typ.colType, mk),
              ),
            )
            memoizeMatrixIR(ctx, child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(ctx, child, requestedType, memo)
        }
      case MatrixExplodeRows(child, path) =>
        def getExplodedField(typ: MatrixType): Type = typ.rowType.queryTyped(path.toList)._1

        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExlosionFieldType =
          try {
            val t = getExplodedField(requestedType)
            preExplosionFieldType match {
              case _: TArray => TArray(t)
              case ts: TSet => ts.copy(elementType = t)
            }
          } catch {
            case _: AnnotationPathException => minimal(preExplosionFieldType)
          }
        val dep = requestedType.copy(rowType =
          unify(
            child.typ.rowType,
            requestedType.rowType.insert(prunedPreExlosionFieldType, path)._1,
          )
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixExplodeCols(child, path) =>
        def getExplodedField(typ: MatrixType): Type = typ.colType.queryTyped(path.toList)._1

        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExplosionFieldType =
          try {
            val t = getExplodedField(requestedType)
            preExplosionFieldType match {
              case _: TArray => TArray(t)
              case ts: TSet => ts.copy(elementType = t)
            }
          } catch {
            case _: AnnotationPathException => minimal(preExplosionFieldType)
          }
        val dep = requestedType.copy(colType =
          unify(
            child.typ.colType,
            requestedType.colType.insert(prunedPreExplosionFieldType, path)._1,
          )
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixRepartition(child, _, _) =>
        memoizeMatrixIR(ctx, child, requestedType, memo)
      case MatrixUnionRows(children) =>
        memoizeMatrixIR(ctx, children.head, requestedType, memo)
        children.tail.foreach(memoizeMatrixIR(
          ctx,
          _,
          requestedType.copy(colType = requestedType.colKeyStruct),
          memo,
        ))
      case MatrixDistinctByRow(child) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(
            child.typ.rowType,
            requestedType.rowType,
            selectKey(child.typ.rowType, child.typ.rowKey),
          ),
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixRowsHead(child, _) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(
            child.typ.rowType,
            requestedType.rowType,
            selectKey(child.typ.rowType, child.typ.rowKey),
          ),
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixColsHead(child, _) => memoizeMatrixIR(ctx, child, requestedType, memo)
      case MatrixRowsTail(child, _) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(
            child.typ.rowType,
            requestedType.rowType,
            selectKey(child.typ.rowType, child.typ.rowKey),
          ),
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixColsTail(child, _) => memoizeMatrixIR(ctx, child, requestedType, memo)
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        val childDep = child.typ.copy(
          key = requestedType.rowKey,
          globalType = unify(
            child.typ.globalType,
            requestedType.globalType,
            TStruct((colsFieldName, TArray(requestedType.colType))),
          ),
          rowType = unify(
            child.typ.rowType,
            requestedType.rowType,
            TStruct((entriesFieldName, TArray(requestedType.entryType))),
          ),
        )
        memoizeTableIR(ctx, child, childDep, memo)
      case MatrixFilterIntervals(child, _, _) =>
        memoizeMatrixIR(
          ctx,
          child,
          requestedType.copy(
            rowKey = child.typ.rowKey,
            rowType = unify(
              child.typ.rowType,
              requestedType.rowType,
              selectKey(child.typ.rowType, child.typ.rowKey),
            ),
          ),
          memo,
        )
      case MatrixToMatrixApply(child, _) => memoizeMatrixIR(ctx, child, child.typ, memo)
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        val globalMapRev = globalMap.map { case (k, v) => (v, k) }
        val colMapRev = colMap.map { case (k, v) => (v, k) }
        val rowMapRev = rowMap.map { case (k, v) => (v, k) }
        val entryMapRev = entryMap.map { case (k, v) => (v, k) }
        val childDep = MatrixType(
          globalType = requestedType.globalType.rename(globalMapRev),
          colType = requestedType.colType.rename(colMapRev),
          rowKey = requestedType.rowKey.map(k => rowMapRev.getOrElse(k, k)),
          colKey = requestedType.colKey.map(k => colMapRev.getOrElse(k, k)),
          rowType = requestedType.rowType.rename(rowMapRev),
          entryType = requestedType.entryType.rename(entryMapRev),
        )
        memoizeMatrixIR(ctx, child, childDep, memo)
      case RelationalLetMatrixTable(name, value, body) =>
        memoizeMatrixIR(ctx, body, requestedType, memo)
        val usages = memo.relationalRefs(name).result()
        memoizeValueIR(ctx, value, unifySeq(value.typ, usages), memo)
    }
  }

  private def memoizeBlockMatrixIR(
    ctx: ExecuteContext,
    bmir: BlockMatrixIR,
    requestedType: BlockMatrixType,
    memo: ComputeMutableState,
  ): Unit = {
    memo.requestedType.bind(bmir, requestedType)
    bmir.children.zipWithIndex.foreach {
      case (mir: MatrixIR, _) => memoizeMatrixIR(ctx, mir, mir.typ, memo)
      case (tir: TableIR, _) => memoizeTableIR(ctx, tir, tir.typ, memo)
      case (bmir: BlockMatrixIR, _) => memoizeBlockMatrixIR(ctx, bmir, bmir.typ, memo)
      case (ir: IR, i) => createTypeStatesAndMemoize(ctx, bmir, i, ir.typ, memo)
    }
  }

  private def memoizeAndGetDep(
    ctx: ExecuteContext,
    ir: TableIR,
    childIdx: Int,
    requestedType: Type,
    memo: ComputeMutableState,
  ): TableType = {
    val Seq(globalState, rowState) =
      createTypeStatesAndMemoize(ctx, ir, childIdx, requestedType, memo)
    val rowType = rowState.newStructType
    val globalType = globalState.newStructType
    TableType(
      key = FastSeq(),
      rowType = rowType,
      globalType = globalType,
    )
  }

  private def memoizeValueIREntryBindings(
    ctx: ExecuteContext,
    ir: MatrixIR,
    childIdx: Int,
    requestedType: Type,
    memo: ComputeMutableState,
  ): MatrixType = {
    val bindings = createTypeStatesAndMemoize(ctx, ir, childIdx, requestedType, memo)

    val globalType = bindings(0).newStructType
    val colType = bindings(1).newStructType
    val rowType = bindings(2).newStructType
    val entryType = bindings(3).newStructType

    if (rowType.hasField(MatrixType.entriesIdentifier))
      throw new RuntimeException(
        s"prune: found dependence on entry array in row binding:\n${Pretty(ctx, ir)}"
      )

    MatrixType(
      rowKey = FastSeq(),
      colKey = FastSeq(),
      globalType = globalType,
      colType = colType,
      rowType = rowType,
      entryType = entryType,
    )
  }

  private def memoizeValueIRRowBindings(
    ctx: ExecuteContext,
    ir: MatrixIR,
    childIdx: Int,
    requestedType: Type,
    memo: ComputeMutableState,
  ): MatrixType = {
    val bindings = createTypeStatesAndMemoize(ctx, ir, childIdx, requestedType, memo)
    val globalType = bindings(0).newStructType
    val rowType = bindings(1).newStructType
    MatrixType(
      rowKey = FastSeq(),
      colKey = FastSeq(),
      globalType = globalType,
      colType = TStruct.empty,
      rowType = rowType,
      entryType = TStruct.empty,
    )
  }

  private def memoizeValueIRColBindings(
    ctx: ExecuteContext,
    ir: MatrixIR,
    childIdx: Int,
    requestedType: Type,
    memo: ComputeMutableState,
  ): MatrixType = {
    val bindings = createTypeStatesAndMemoize(ctx, ir, childIdx, requestedType, memo)
    val globalType = bindings(0).newStructType
    val colType = bindings(1).newStructType
    MatrixType(
      rowKey = FastSeq(),
      colKey = FastSeq(),
      globalType = globalType,
      colType = colType,
      rowType = TStruct.empty,
      entryType = TStruct.empty,
    )
  }

  private def createTypeStatesAndMemoize(
    ctx: ExecuteContext,
    ir: BaseIR,
    childIdx: Int,
    requestedType: Type,
    memo: ComputeMutableState,
  ): IndexedSeq[TypeState] =
    createTypeStatesAndMemoize(
      ctx,
      ir,
      childIdx,
      requestedType,
      memo,
      BindingEnv.empty.createAgg.createScan,
    )

  private def createTypeStatesAndMemoize(
    ctx: ExecuteContext,
    ir: BaseIR,
    childIdx: Int,
    requestedType: Type,
    memo: ComputeMutableState,
    env: BindingEnv[TypeState],
  ): IndexedSeq[TypeState] = {
    val bindings = Bindings.get(ir, childIdx)
    val bindingsStates = bindings.map((_, typ) => TypeState(typ))
    memoizeValueIR(
      ctx,
      ir.getChild(childIdx).asInstanceOf[IR],
      requestedType,
      memo,
      env.extend(bindingsStates),
    )
    bindingsStates.all.map(_._2)
  }

  /** This function does *not* necessarily bind each child node in `memo`. Known dead code is not
    * memoized. For instance:
    *
    * ir = MakeStruct(Seq("a" -> (child1), "b" -> (child2))) requestedType = TStruct("a" -> (reqType
    * of a))
    *
    * In the above, `child2` will not be memoized because `ir` does not require any of the "b"
    * dependencies in order to create its own requested type, which only contains "a".
    */
  def memoizeValueIR(
    ctx: ExecuteContext,
    ir: IR,
    requestedType: Type,
    memo: ComputeMutableState,
    env: BindingEnv[TypeState] = BindingEnv.empty.createAgg.createScan,
  ): Unit = {
    def recurMax(ir: IR, childIdx: Int): Unit =
      recur(ir, childIdx, ir.getChild(childIdx).asInstanceOf[IR].typ)

    def recurMaxWithBindings(ir: IR, childIdx: Int): IndexedSeq[TypeState] =
      recurWithBindings(ir, childIdx, ir.getChild(childIdx).asInstanceOf[IR].typ)

    def recurMin(ir: IR, childIdx: Int): Unit =
      recur(ir, childIdx, minimal(ir.getChild(childIdx).asInstanceOf[IR].typ))

    def recurWithBindings(ir: IR, childIdx: Int, requestedType: Type): IndexedSeq[TypeState] =
      createTypeStatesAndMemoize(ctx, ir, childIdx, requestedType, memo, env)

    def recur(ir: IR, childIdx: Int, requestedType: Type): Unit = {
      val bindings = Bindings.get(ir, childIdx)
      val bindingsStates = bindings.map((_, typ) => TypeState(typ))
      memoizeValueIR(
        ctx,
        ir.getChild(childIdx).asInstanceOf[IR],
        requestedType,
        memo,
        env.extend(bindingsStates),
      )
    }

    def recurMaxWithTypeStates(ir: IR, childIdx: Int, bindingsMap: mutable.Map[Name, TypeState])
      : Unit =
      recurWithTypeStates(ir, childIdx, ir.getChild(childIdx).asInstanceOf[IR].typ, bindingsMap)

    def recurWithTypeStates(
      ir: IR,
      childIdx: Int,
      requestedType: Type,
      bindingsMap: mutable.Map[Name, TypeState],
    ): Unit = {
      val bindings = Bindings.get(ir, childIdx).map { (name, typ) =>
        val state = bindingsMap.getOrElseUpdate(name, TypeState(typ))
        assert(typ == state.origType)
        state
      }
      memoizeValueIR(
        ctx,
        ir.getChild(childIdx).asInstanceOf[IR],
        requestedType,
        memo,
        env.extend(bindings),
      )
    }

    memo.requestedType.bind(ir, requestedType)
    ir match {
      case ir: IsNA =>
        recurMin(ir, 0)

      case CastRename(v, _typ) =>
        def loop(reqType: Type, castType: Type, baseType: Type): Type = {
          ((reqType, castType, baseType): @unchecked) match {
            case (TStruct(reqFields), cast: TStruct, base: TStruct) =>
              TStruct(reqFields.map { f =>
                val idx = cast.fieldIdx(f.name)
                Field(base.fieldNames(idx), loop(f.typ, cast.types(idx), base.types(idx)), f.index)
              })
            case (TTuple(req), TTuple(cast), TTuple(base)) =>
              assert(base.length == cast.length)
              val castFields = cast.map(f => f.index -> f.typ).toMap
              val baseFields = base.map(f => f.index -> f.typ).toMap
              TTuple(req.map { f =>
                TupleField(f.index, loop(f.typ, castFields(f.index), baseFields(f.index)))
              })
            case (TArray(req), TArray(cast), TArray(base)) =>
              TArray(loop(req, cast, base))
            case (TSet(req), TSet(cast), TSet(base)) =>
              TSet(loop(req, cast, base))
            case (TDict(reqK, reqV), TDict(castK, castV), TDict(baseK, baseV)) =>
              TDict(loop(reqK, castK, baseK), loop(reqV, castV, baseV))
            case (TInterval(req), TInterval(cast), TInterval(base)) =>
              TInterval(loop(req, cast, base))
            case _ => reqType
          }
        }

        recur(ir, 0, loop(requestedType, _typ, v.typ))

      case ir: If =>
        recurMax(ir, 0)
        recur(ir, 1, requestedType)
        recur(ir, 2, requestedType)

      case Switch(_, _, cases) =>
        recurMax(ir, 0)
        recur(ir, 1, requestedType)
        cases.indices.foreach(i => recur(ir, i + 2, requestedType))

      case Coalesce(values) =>
        values.indices.foreach(recur(ir, _, requestedType))

      case Consume(_) =>
        recurMax(ir, 0)

      case Block(bindings, _) =>
        val typeStates = is.hail.utils.compat.mutable.AnyRefMap.empty[Name, TypeState]
        recurWithTypeStates(ir, bindings.length, requestedType, typeStates)
        for (i <- bindings.indices.reverse)
          recurWithTypeStates(ir, i, typeStates(bindings(i).name).newType, typeStates)

      case Ref(name, _) =>
//        env.eval.lookupOption(name).foreach(_.union(requestedType))
        env.eval(name).union(requestedType): Unit

      case RelationalLet(name, value, _) =>
        recur(ir, 1, requestedType)
        val usages = memo.relationalRefs(name).result()
        recur(ir, 0, unifySeq(value.typ, usages))

      case RelationalRef(name, _) =>
        memo.relationalRefs(name) += requestedType

      case MakeArray(args, _) =>
        val eltType = TIterable.elementType(requestedType)
        args.indices.foreach(recur(ir, _, eltType))

      case MakeStream(args, _, _) =>
        val eltType = TIterable.elementType(requestedType)
        args.indices.foreach(recur(ir, _, eltType))

      case ir: ArrayRef =>
        recur(ir, 0, TArray(requestedType))
        recurMax(ir, 1)

      case ir: ArrayLen =>
        recurMin(ir, 0)

      case ir: StreamTake =>
        recur(ir, 0, requestedType)
        recurMax(ir, 1)

      case ir: StreamDrop =>
        recur(ir, 0, requestedType)
        recurMax(ir, 1)

      case StreamWhiten(a, newChunk, prevWindow, _, _, _, _, _) =>
        val matType = TNDArray(TFloat64, Nat(2))
        val unifiedStructType = unify(
          a.typ.asInstanceOf[TStream].elementType,
          requestedType.asInstanceOf[TStream].elementType,
          TStruct((newChunk, matType), (prevWindow, matType)),
        )
        recur(ir, 0, TStream(unifiedStructType))

      case ir: StreamGrouped =>
        recur(ir, 0, TIterable.elementType(requestedType))
        recurMax(ir, 1)

      case StreamGroupByKey(a, key, _) =>
        val reqStructT = tcoerce[TStruct](
          tcoerce[TStream](tcoerce[TStream](requestedType).elementType).elementType
        )
        val origStructT = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
        recur(
          ir,
          0,
          TStream(unify(origStructT, reqStructT, selectKey(origStructT, key))),
        )

      case StreamZip(as, _, _, behavior, _) =>
        val bodyBindings = recurWithBindings(ir, as.length, TIterable.elementType(requestedType))
        if (behavior == ArrayZipBehavior.AssumeSameLength && bodyBindings.forall(_.isUndefined)) {
          recurMin(ir, 0)
        } else {
          as.indices.foreach { i =>
            val state = bodyBindings(i)
            if (behavior != ArrayZipBehavior.AssumeSameLength || !state.isUndefined) {
              recur(ir, i, TStream(state.newType))
            }
          }
        }

      case StreamZipJoin(as, key, _, _, _) =>
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val Seq(_, valsState) = recurWithBindings(ir, as.length, requestedEltType)
        val childRequestedEltType = elementType(valsState.requireFieldsInElt(key).newType)
        as.indices.foreach(recur(ir, _, TStream(childRequestedEltType)))

      case StreamZipJoinProducers(_, _, _, key, _, _, _) =>
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val Seq(_, valsState) = recurWithBindings(ir, 2, requestedEltType)
        val producerRequestedEltType = elementType(valsState.requireFieldsInElt(key).newType)
        val Seq(ctxState) = recurWithBindings(ir, 1, TStream(producerRequestedEltType))
        recur(ir, 0, TArray(ctxState.newType))

      case StreamMultiMerge(as, key) =>
        val eltType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val childRequestedEltType = unify(eltType, requestedEltType, selectKey(eltType, key))
        as.indices.foreach(recur(ir, _, TStream(childRequestedEltType)))

      case _: StreamFilter | _: StreamTakeWhile | _: StreamDropWhile =>
        val Seq(eltState) = recurMaxWithBindings(ir, 1)
        val valueType = eltState.union(TIterable.elementType(requestedType)).newType
        recur(ir, 0, TStream(valueType))

      case StreamFlatMap(_, _, _) =>
        val Seq(eltState) = recurWithBindings(ir, 1, requestedType)
        recur(ir, 0, TStream(eltState.newType))

      case _: StreamFold | _: StreamScan =>
        val Seq(_, valueState) = recurMaxWithBindings(ir, 2)
        recurMax(ir, 1)
        recur(ir, 0, TStream(valueState.newType))

      case StreamFold2(_, accum, valueName, seq, _) =>
        recur(ir, 2 * accum.length + 1, requestedType)
        val seqBindings = is.hail.utils.compat.mutable.AnyRefMap.empty[Name, TypeState]
        seq.indices.foreach(i => recurMaxWithTypeStates(ir, accum.length + 1 + i, seqBindings))
        accum.indices.foreach(i => recurMax(ir, i + 1))
        recur(ir, 0, TStream(seqBindings(valueName).newType))

      case StreamJoinRightDistinct(_, _, lKey, rKey, _, _, _, _) =>
        val Seq(lState, rState) = recurWithBindings(ir, 2, TIterable.elementType(requestedType))
        val lRequested = lState.requireFields(lKey).newType
        val rRequested = rState.requireFields(rKey).newType
        recur(ir, 0, TStream(lRequested))
        recur(ir, 1, TStream(rRequested))

      case StreamLeftIntervalJoin(_, _, keyFieldName, intervalFieldName, _, _, _) =>
        val Seq(lState, rState) = recurWithBindings(ir, 2, elementType(requestedType))
        val lRequestedType = lState.requireFields(FastSeq(keyFieldName)).newType
        val rEltType = elementType(rState.origType).asInstanceOf[TStruct]
        val rRequestedType =
          rState.union(TArray(selectKey(rEltType, FastSeq(intervalFieldName)))).newType
        recur(ir, 0, TStream(lRequestedType))
        recur(ir, 1, TStream(elementType(rRequestedType)))

      case ArraySort(_, _, _, _) =>
        val Seq(lState, rState) = recurMaxWithBindings(ir, 1)
        val requestedElementType = lState
          .union(rState.newType)
          .union(TIterable.elementType(requestedType))
          .newType
        recur(ir, 0, TStream(requestedElementType))

      case ArrayMaximalIndependentSet(_, tiebreaker) =>
        if (tiebreaker.nonEmpty) recurMax(ir, 1)
        recurMax(ir, 0)

      case StreamFor(_, _, _) =>
        assert(requestedType == TVoid)
        val Seq(eltState) = recurMaxWithBindings(ir, 1)
        recur(ir, 0, TStream(eltState.newType))

      case MakeNDArray(data, _, _, _) =>
        val elementType = requestedType.asInstanceOf[TNDArray].elementType
        val dataType =
          if (data.typ.isInstanceOf[TArray]) TArray(elementType)
          else TStream(elementType)
        recur(ir, 0, dataType)
        recurMax(ir, 1)
        recurMax(ir, 2)

      case NDArrayMap(nd, _, _) =>
        val Seq(eltState) =
          recurWithBindings(ir, 1, requestedType.asInstanceOf[TNDArray].elementType)
        val eltType =
          nd.typ.asInstanceOf[TNDArray].copy(elementType = eltState.newType)
        recur(ir, 0, eltType)

      case NDArrayMap2(left, right, _, _, _, _) =>
        val Seq(lState, rState) =
          recurWithBindings(ir, 2, requestedType.asInstanceOf[TNDArray].elementType)
        recur(ir, 0, left.typ.asInstanceOf[TNDArray].copy(elementType = lState.newType))
        recur(ir, 1, right.typ.asInstanceOf[TNDArray].copy(elementType = rState.newType))

      case AggExplode(_, _, _, _) =>
        val Seq(eltState) = recurWithBindings(ir, 1, requestedType)
        recur(ir, 0, TStream(eltState.newType))

      case AggFilter(_, _, _) =>
        recur(ir, 1, requestedType)
        recurMax(ir, 0)

      case AggGroupBy(_, _, _) =>
        val tdict = requestedType.asInstanceOf[TDict]
        recur(ir, 1, tdict.valueType)
        recur(ir, 0, tdict.keyType)

      case AggArrayPerElement(_, _, _, _, knownLength, _) =>
        val Seq(eltState, _) = recurWithBindings(ir, 1, TIterable.elementType(requestedType))
        recur(ir, 0, TArray(eltState.newType))
        if (knownLength.nonEmpty) recurMax(ir, 2)

      case a @ (_: ApplyAggOp | _: ApplyScanOp) =>
        val (initOpArgs, seqOpArgs, op) = a match {
          case ApplyAggOp(initOpArgs, seqOpArgs, op) =>
            (initOpArgs.map(_.typ), seqOpArgs.map(_.typ), op)
          case ApplyScanOp(initOpArgs, seqOpArgs, op) =>
            (initOpArgs.map(_.typ), seqOpArgs.map(_.typ), op)
        }

        val prunedSeqOpArgs = AggOp.prune(op, seqOpArgs, requestedType)
        initOpArgs.zipWithIndex.foreach { case (req, i) =>
          recur(ir, i, req)
        }
        prunedSeqOpArgs.zipWithIndex.foreach { case (req, i) =>
          recur(ir, initOpArgs.length + i, req)
        }

      case ir: AggFold =>
        (0 until 3).foreach(i => recurMax(ir, i))

      case StreamAgg(_, _, _) =>
        val Seq(eltState) = recurWithBindings(ir, 1, requestedType)
        recur(ir, 0, TStream(eltState.newType))

      case _: StreamMap | _: StreamAggScan =>
        val Seq(eltState) = recurWithBindings(ir, 1, TIterable.elementType(requestedType))
        recur(ir, 0, TStream(eltState.newType))

      case ir: RunAgg =>
        recurMax(ir, 0)
        recur(ir, 1, requestedType)

      case RunAggScan(_, name, _, _, _, _) =>
        val bindings = is.hail.utils.compat.mutable.AnyRefMap.empty[Name, TypeState]

        recurWithTypeStates(ir, 3, TIterable.elementType(requestedType), bindings)
        recurMaxWithTypeStates(ir, 2, bindings)
        recurMax(ir, 1)
        recur(ir, 0, TStream(bindings(name).newType))

      case MakeStruct(fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        fields.view.zipWithIndex.foreach { case ((fname, _), i) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          sType.selfField(fname).foreach(f => recur(ir, i, f.typ))
        }

      case InsertFields(old, fields, _) =>
        val sType = requestedType.asInstanceOf[TStruct]
        val rightDep = sType.typeAfterSelectNames(sType.fieldNames.intersect(fields.map(_._1)))
        val leftDep = TStruct(
          old.typ.asInstanceOf[TStruct]
            .fields
            .flatMap { f =>
              if (rightDep.hasField(f.name))
                Some(f.name -> minimal(f.typ))
              else
                sType.selfField(f.name).map(f.name -> _.typ)
            }: _*
        )
        recur(ir, 0, leftDep)
        fields.view.zipWithIndex.foreach { case ((fname, _), i) =>
          rightDep.selfField(fname).foreach(f => recur(ir, i + 1, f.typ))
        }

      case SelectFields(old, _) =>
        val sType = requestedType.asInstanceOf[TStruct]
        val oldReqType = TStruct(old.typ.asInstanceOf[TStruct]
          .fieldNames
          .flatMap(fn => sType.selfField(fn).map(fd => (fd.name, fd.typ))): _*)
        recur(ir, 0, oldReqType)

      case GetField(_, name) =>
        recur(ir, 0, TStruct(name -> requestedType))

      case MakeTuple(fields) =>
        val tType = requestedType.asInstanceOf[TTuple]
        fields.view.zipWithIndex.foreach { case ((fieldIdx, _), childIdx) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          tType.fieldIndex.get(fieldIdx).foreach(idx => recur(ir, childIdx, tType.types(idx)))
        }
      case GetTupleElement(_, idx) =>
        val tupleDep = TTuple(FastSeq(TupleField(idx, requestedType)))
        recur(ir, 0, tupleDep)

      case ir: ConsoleLog =>
        recur(ir, 0, TString)
        recurMax(ir, 1)

      case MatrixCount(child) =>
        memoizeMatrixIR(ctx, child, minimal(child.typ), memo)

      case TableCount(child) =>
        memoizeTableIR(ctx, child, minimal(child.typ), memo)

      case TableGetGlobals(child) =>
        val childReqType = minimal(child.typ).copy(globalType = requestedType.asInstanceOf[TStruct])
        memoizeTableIR(ctx, child, childReqType, memo)

      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        memoizeTableIR(
          ctx,
          child,
          TableType(
            key = child.typ.key,
            rowType = unify(
              child.typ.rowType,
              rStruct.selfField("rows").map(
                _.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
              ).getOrElse(TStruct.empty),
            ),
            globalType =
              rStruct.selfField("global").map(_.typ.asInstanceOf[TStruct]).getOrElse(TStruct.empty),
          ),
          memo,
        )

      case TableToValueApply(child, _) =>
        memoizeTableIR(ctx, child, child.typ, memo)

      case MatrixToValueApply(child, _) =>
        memoizeMatrixIR(ctx, child, child.typ, memo)

      case BlockMatrixToValueApply(child, _) =>
        memoizeBlockMatrixIR(ctx, child, child.typ, memo)

      case TableAggregate(child, _) =>
        val Seq(globalState, rowState) = recurMaxWithBindings(ir, 1)
        val dep = TableType(
          key = child.typ.key,
          rowType = rowState.requireFields(child.typ.key).newStructType,
          globalType = globalState.newStructType,
        )
        memoizeTableIR(ctx, child, dep, memo)

      case MatrixAggregate(child, _) =>
        val Seq(globalState, colState, rowState, entryState) = recurMaxWithBindings(ir, 1)
        val dep = MatrixType(
          rowKey = child.typ.rowKey,
          colKey = FastSeq(),
          rowType = rowState.requireFields(child.typ.rowKey).newStructType,
          entryType = entryState.newStructType,
          colType = colState.newStructType,
          globalType = globalState.newStructType,
        )
        memoizeMatrixIR(ctx, child, dep, memo)

      case TailLoop(_, params, _, _) =>
        val paramStates = recurMaxWithBindings(ir, params.length)
        paramStates.view.zipWithIndex.take(params.length).foreach { case (paramState, i) =>
          recur(ir, i, paramState.newType)
        }

      case CollectDistributedArray(_, _, _, _, _, _, _, _) =>
        recur(ir, 3, TString)
        val Seq(contextState, globalState) =
          recurWithBindings(ir, 2, requestedType.asInstanceOf[TArray].elementType)
        recur(ir, 1, globalState.newType)
        recur(ir, 0, TStream(contextState.newType))

      case _: IR =>
        ir.children.zipWithIndex.foreach {
          case (mir: MatrixIR, _) =>
            memoizeMatrixIR(ctx, mir, mir.typ, memo)
          case (tir: TableIR, _) =>
            memoizeTableIR(ctx, tir, tir.typ, memo)
          case (_: BlockMatrixIR, _) => // NOTE Currently no BlockMatrixIRs would have dead fields
          case (_: IR, i) =>
            recurMax(ir, i)
        }
    }
  }

  def rebuild(
    ctx: ExecuteContext,
    tir: TableIR,
    memo: RebuildMutableState,
  ): TableIR = {
    val requestedType = memo.requestedType.lookup(tir).asInstanceOf[TableType]
    tir match {
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        TableParallelize(
          upcast(
            ctx,
            rebuildIR(ctx, rowsAndGlobal, BindingEnv.empty, memo),
            memo.requestedType.lookup(rowsAndGlobal).asInstanceOf[TStruct],
          ),
          nPartitions,
        )

      case TableGen(contexts, globals, cname, gname, body, partitioner, errorId) =>
        val newContexts = rebuildIR(ctx, contexts, BindingEnv.empty, memo)
        val newGlobals = rebuildIR(ctx, globals, BindingEnv.empty, memo)
        val bodyEnv = Env(cname -> TIterable.elementType(newContexts.typ), gname -> newGlobals.typ)
        TableGen(
          contexts = newContexts,
          globals = newGlobals,
          cname = cname,
          gname = gname,
          body = rebuildIR(ctx, body, BindingEnv(bodyEnv), memo),
          partitioner.coarsen(requestedType.key.length),
          errorId,
        )

      case TableRead(typ, dropRows, tr) =>
        // FIXME: remove this when all readers know how to read without keys
        val requestedTypeWithKey = TableType(
          key = typ.key,
          rowType = unify(typ.rowType, selectKey(typ.rowType, typ.key), requestedType.rowType),
          globalType = requestedType.globalType,
        )
        TableRead(requestedTypeWithKey, dropRows, tr)
      case TableFilter(child, pred) =>
        val child2 = rebuild(ctx, child, memo)
        val pred2 = rebuildIR(ctx, pred, BindingEnv(child2.typ.rowEnv), memo)
        TableFilter(child2, pred2)
      case TableMapPartitions(child, gName, pName, body, requestedKey, allowedOverlap) =>
        val child2 = rebuild(ctx, child, memo)
        val body2 = rebuildIR(
          ctx,
          body,
          BindingEnv(Env(
            gName -> child2.typ.globalType,
            pName -> TStream(child2.typ.rowType),
          )),
          memo,
        )
        val body2ElementType = TIterable.elementType(body2.typ).asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.key.exists(k => !body2ElementType.hasField(k)))
          TableKeyBy(child2, child2.typ.key.takeWhile(body2ElementType.hasField))
        else
          child2
        val childKeyLen = child2Keyed.typ.key.length
        require(requestedKey <= childKeyLen)
        TableMapPartitions(
          child2Keyed,
          gName,
          pName,
          body2,
          requestedKey,
          math.min(allowedOverlap, childKeyLen),
        )
      case TableMapRows(child, newRow) =>
        val child2 = rebuild(ctx, child, memo)
        val newRow2 = rebuildIR(
          ctx,
          newRow,
          BindingEnv(child2.typ.rowEnv, scan = Some(child2.typ.rowEnv)),
          memo,
        )
        val newRowType = newRow2.typ.asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.key.exists(k => !newRowType.hasField(k))) {
          val upcastKey = child2.typ.key.takeWhile(newRowType.hasField)
          assert(upcastKey.startsWith(requestedType.key))
          TableKeyBy(child2, upcastKey)
        } else
          child2
        TableMapRows(child2Keyed, newRow2)
      case TableMapGlobals(child, newGlobals) =>
        val child2 = rebuild(ctx, child, memo)
        TableMapGlobals(child2, rebuildIR(ctx, newGlobals, BindingEnv(child2.typ.globalEnv), memo))
      case TableKeyBy(child, _, isSorted) =>
        var child2 = rebuild(ctx, child, memo)
        val keys2 = requestedType.key
        // fully upcast before shuffle
        if (!isSorted && keys2.nonEmpty)
          child2 = upcastTable(
            ctx,
            child2,
            memo.requestedType.lookup(child).asInstanceOf[TableType],
            upcastGlobals = false,
          )
        TableKeyBy(child2, keys2, isSorted)
      case TableOrderBy(child, sortFields) =>
        val child2 =
          if (
            sortFields.forall(_.sortOrder == Ascending) && child.typ.key.startsWith(
              sortFields.map(_.field)
            )
          )
            rebuild(ctx, child, memo)
          else {
            // fully upcast before shuffle
            upcastTable(
              ctx,
              rebuild(ctx, child, memo),
              memo.requestedType.lookup(child).asInstanceOf[TableType],
              upcastGlobals = false,
            )
          }
        TableOrderBy(child2, sortFields)
      case TableLeftJoinRightDistinct(left, right, root) =>
        if (requestedType.rowType.hasField(root))
          TableLeftJoinRightDistinct(rebuild(ctx, left, memo), rebuild(ctx, right, memo), root)
        else
          rebuild(ctx, left, memo)
      case TableIntervalJoin(left, right, root, product) =>
        if (requestedType.rowType.hasField(root))
          TableIntervalJoin(rebuild(ctx, left, memo), rebuild(ctx, right, memo), root, product)
        else
          rebuild(ctx, left, memo)
      case TableMultiWayZipJoin(children, fieldName, globalName) =>
        val rebuilt = children.map(c => rebuild(ctx, c, memo))
        val upcasted = rebuilt.map { t =>
          upcastTable(ctx, t, memo.requestedType.lookup(children(0)).asInstanceOf[TableType])
        }
        TableMultiWayZipJoin(upcasted, fieldName, globalName)
      case TableAggregateByKey(child, expr) =>
        val child2 = rebuild(ctx, child, memo)
        TableAggregateByKey(
          child2,
          rebuildIR(
            ctx,
            expr,
            BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)),
            memo,
          ),
        )
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val child2 = rebuild(ctx, child, memo)
        val expr2 = rebuildIR(
          ctx,
          expr,
          BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)),
          memo,
        )
        val newKey2 = rebuildIR(ctx, newKey, BindingEnv(child2.typ.rowEnv), memo)
        TableKeyByAndAggregate(child2, expr2, newKey2, nPartitions, bufferSize)
      case TableRename(child, rowMap, globalMap) =>
        val child2 = rebuild(ctx, child, memo)
        TableRename(
          child2,
          rowMap.view.filterKeys(child2.typ.rowType.hasField).toMap,
          globalMap.view.filterKeys(child2.typ.globalType.hasField).toMap,
        )
      case TableUnion(children) =>
        val requestedType = memo.requestedType.lookup(tir).asInstanceOf[TableType]
        val rebuilt = children.map { c =>
          upcastTable(ctx, rebuild(ctx, c, memo), requestedType, upcastGlobals = false)
        }
        TableUnion(rebuilt)
      case RelationalLetTable(name, value, body) =>
        val value2 = rebuildIR(ctx, value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLetTable(name, value2, rebuild(ctx, body, memo))
      case BlockMatrixToTableApply(bmir, aux, function) =>
        val bmir2 = rebuild(ctx, bmir, memo)
        val aux2 = rebuildIR(ctx, aux, BindingEnv.empty, memo)
        BlockMatrixToTableApply(bmir2, aux2, function)
      case _ => tir.mapChildren {
          // IR should be a match error - all nodes with child value IRs should have a rule
          case childT: TableIR => rebuild(ctx, childT, memo)
          case childM: MatrixIR => rebuild(ctx, childM, memo)
          case childBm: BlockMatrixIR => rebuild(ctx, childBm, memo)
        }.asInstanceOf[TableIR]
    }
  }

  def rebuild(
    ctx: ExecuteContext,
    mir: MatrixIR,
    memo: RebuildMutableState,
  ): MatrixIR = {
    val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
    mir match {
      case MatrixRead(typ, dropCols, dropRows, reader) =>
        // FIXME: remove this when all readers know how to read without keys
        val requestedTypeWithKeys = MatrixType(
          rowKey = typ.rowKey,
          colKey = typ.colKey,
          rowType = unify(typ.rowType, selectKey(typ.rowType, typ.rowKey), requestedType.rowType),
          entryType = requestedType.entryType,
          colType = unify(typ.colType, selectKey(typ.colType, typ.colKey), requestedType.colType),
          globalType = requestedType.globalType,
        )
        MatrixRead(requestedTypeWithKeys, dropCols, dropRows, reader)
      case MatrixFilterCols(child, pred) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixFilterCols(child2, rebuildIR(ctx, pred, BindingEnv(child2.typ.colEnv), memo))
      case MatrixFilterRows(child, pred) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixFilterRows(child2, rebuildIR(ctx, pred, BindingEnv(child2.typ.rowEnv), memo))
      case MatrixFilterEntries(child, pred) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixFilterEntries(child2, rebuildIR(ctx, pred, BindingEnv(child2.typ.entryEnv), memo))
      case MatrixMapEntries(child, newEntries) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixMapEntries(child2, rebuildIR(ctx, newEntries, BindingEnv(child2.typ.entryEnv), memo))
      case MatrixMapRows(child, newRow) =>
        val child2 = rebuild(ctx, child, memo)
        val newRow2 = rebuildIR(
          ctx,
          newRow,
          BindingEnv(
            child2.typ.rowEnv,
            agg = Some(child2.typ.entryEnv),
            scan = Some(child2.typ.rowEnv),
          ),
          memo,
        )
        val newRowType = newRow2.typ.asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.rowKey.exists(k => !newRowType.hasField(k)))
          MatrixKeyRowsBy(child2, child2.typ.rowKey.takeWhile(newRowType.hasField))
        else
          child2
        MatrixMapRows(child2Keyed, newRow2)
      case MatrixMapCols(child, newCol, newKey) =>
        val child2 = rebuild(ctx, child, memo)
        val newCol2 = rebuildIR(
          ctx,
          newCol,
          BindingEnv(
            child2.typ.colEnv,
            agg = Some(child2.typ.entryEnv),
            scan = Some(child2.typ.colEnv),
          ),
          memo,
        )
        val newColType = newCol2.typ.asInstanceOf[TStruct]
        val newKey2 = newKey match {
          case Some(nk) => Some(nk.takeWhile(newColType.hasField))
          case None => if (child2.typ.colKey.exists(k => !newColType.hasField(k)))
              Some(child2.typ.colKey.takeWhile(newColType.hasField))
            else
              None
        }
        MatrixMapCols(child2, newCol2, newKey2)
      case MatrixMapGlobals(child, newGlobals) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixMapGlobals(child2, rebuildIR(ctx, newGlobals, BindingEnv(child2.typ.globalEnv), memo))
      case MatrixKeyRowsBy(child, keys, isSorted) =>
        val child2 = rebuild(ctx, child, memo)
        val keys2 = keys.takeWhile(child2.typ.rowType.hasField)
        MatrixKeyRowsBy(child2, keys2, isSorted)
      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixAggregateRowsByKey(
          child2,
          rebuildIR(
            ctx,
            entryExpr,
            BindingEnv(child2.typ.colEnv, agg = Some(child2.typ.entryEnv)),
            memo,
          ),
          rebuildIR(
            ctx,
            rowExpr,
            BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)),
            memo,
          ),
        )
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixAggregateColsByKey(
          child2,
          rebuildIR(
            ctx,
            entryExpr,
            BindingEnv(child2.typ.rowEnv, agg = Some(child2.typ.entryEnv)),
            memo,
          ),
          rebuildIR(
            ctx,
            colExpr,
            BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.colEnv)),
            memo,
          ),
        )
      case MatrixUnionRows(children) =>
        val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
        val firstChild =
          upcast(ctx, rebuild(ctx, children.head, memo), requestedType, upcastGlobals = false)
        val remainingChildren = children.tail.map { child =>
          upcast(
            ctx,
            rebuild(ctx, child, memo),
            requestedType.copy(colType = requestedType.colKeyStruct),
            upcastGlobals = false,
          )
        }
        MatrixUnionRows(firstChild +: remainingChildren)
      case MatrixUnionCols(left, right, joinType) =>
        val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
        val left2 = rebuild(ctx, left, memo)
        val right2 = rebuild(ctx, right, memo)

        if (
          left2.typ.colType == right2.typ.colType && left2.typ.entryType == right2.typ.entryType
        ) {
          MatrixUnionCols(
            left2,
            right2,
            joinType,
          )
        } else {
          MatrixUnionCols(
            upcast(ctx, left2, requestedType, upcastRows = false, upcastGlobals = false),
            upcast(ctx, right2, requestedType, upcastRows = false, upcastGlobals = false),
            joinType,
          )
        }
      case MatrixAnnotateRowsTable(child, table, root, product) =>
        // if the field is not used, this node can be elided entirely
        if (!requestedType.rowType.hasField(root))
          rebuild(ctx, child, memo)
        else {
          val child2 = rebuild(ctx, child, memo)
          val table2 = rebuild(ctx, table, memo)
          MatrixAnnotateRowsTable(child2, table2, root, product)
        }
      case MatrixAnnotateColsTable(child, table, uid) =>
        // if the field is not used, this node can be elided entirely
        if (!requestedType.colType.hasField(uid))
          rebuild(ctx, child, memo)
        else {
          val child2 = rebuild(ctx, child, memo)
          val table2 = rebuild(ctx, table, memo)
          MatrixAnnotateColsTable(child2, table2, uid)
        }
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixRename(
          child2,
          globalMap.view.filterKeys(child2.typ.globalType.hasField).toMap,
          colMap.view.filterKeys(child2.typ.colType.hasField).toMap,
          rowMap.view.filterKeys(child2.typ.rowType.hasField).toMap,
          entryMap.view.filterKeys(child2.typ.entryType.hasField).toMap,
        )
      case RelationalLetMatrixTable(name, value, body) =>
        val value2 = rebuildIR(ctx, value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLetMatrixTable(name, value2, rebuild(ctx, body, memo))
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        CastTableToMatrix(
          rebuild(ctx, child, memo),
          entriesFieldName,
          colsFieldName,
          requestedType.colKey,
        )
      case _ => mir.mapChildren {
          // IR should be a match error - all nodes with child value IRs should have a rule
          case childT: TableIR => rebuild(ctx, childT, memo)
          case childM: MatrixIR => rebuild(ctx, childM, memo)
        }.asInstanceOf[MatrixIR]
    }
  }

  def rebuild(
    ctx: ExecuteContext,
    bmir: BlockMatrixIR,
    memo: RebuildMutableState,
  ): BlockMatrixIR =
    bmir.mapChildren {
      case tir: TableIR => rebuild(ctx, tir, memo)
      case mir: MatrixIR => rebuild(ctx, mir, memo)
      case ir: IR => rebuildIR(ctx, ir, BindingEnv.empty[Type], memo)
      case bmir: BlockMatrixIR => rebuild(ctx, bmir, memo)
    }.asInstanceOf[BlockMatrixIR]

  def rebuildIR(
    ctx: ExecuteContext,
    ir: IR,
    env: BindingEnv[Type],
    memo: RebuildMutableState,
  ): IR = {
    val requestedType = memo.requestedType.lookup(ir).asInstanceOf[Type]
    ir match {
      case NA(_) => NA(requestedType)
      case CastRename(v, _typ) =>
        val v2 = rebuildIR(ctx, v, env, memo)

        def recur(rebuildType: Type, castType: Type, baseType: Type): Type = {
          ((rebuildType, castType, baseType): @unchecked) match {
            case (TStruct(rebFields), cast: TStruct, base: TStruct) =>
              TStruct(rebFields.map { f =>
                val idx = base.fieldIdx(f.name)
                Field(cast.fieldNames(idx), recur(f.typ, cast.types(idx), base.types(idx)), f.index)
              })
            case (TTuple(reb), TTuple(cast), TTuple(base)) =>
              assert(base.length == cast.length)
              val castFields = cast.map(f => f.index -> f.typ).toMap
              val baseFields = base.map(f => f.index -> f.typ).toMap
              TTuple(reb.map { f =>
                TupleField(f.index, recur(f.typ, castFields(f.index), baseFields(f.index)))
              })
            case (TArray(reb), TArray(cast), TArray(base)) =>
              TArray(recur(reb, cast, base))
            case (TSet(reb), TSet(cast), TSet(base)) =>
              TSet(recur(reb, cast, base))
            case (TDict(rebK, rebV), TDict(castK, castV), TDict(baseK, baseV)) =>
              TDict(recur(rebK, castK, baseK), recur(rebV, castV, baseV))
            case (TInterval(reb), TInterval(cast), TInterval(base)) =>
              TInterval(recur(reb, cast, base))
            case _ => rebuildType
          }
        }

        CastRename(v2, recur(v2.typ, _typ, v.typ))
      case If(cond, cnsq, alt) =>
        val cond2 = rebuildIR(ctx, cond, env, memo)
        val cnsq2 = rebuildIR(ctx, cnsq, env, memo)
        val alt2 = rebuildIR(ctx, alt, env, memo)

        if (cnsq2.typ == alt2.typ)
          If(cond2, cnsq2, alt2)
        else
          If(cond2, upcast(ctx, cnsq2, requestedType), upcast(ctx, alt2, requestedType))
      case Coalesce(values) =>
        val values2 = values.map(rebuildIR(ctx, _, env, memo))
        require(values2.nonEmpty)
        if (values2.forall(_.typ == values2.head.typ))
          Coalesce(values2)
        else
          Coalesce(values2.map(upcast(ctx, _, requestedType)))
      case Ref(name, t) =>
        Ref(name, env.eval.lookupOption(name).getOrElse(t))
      case RelationalLet(name, value, body) =>
        val value2 = rebuildIR(ctx, value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLet(name, value2, rebuildIR(ctx, body, env, memo))
      case RelationalRef(name, _) => RelationalRef(name, memo.relationalRefs(name))
      case MakeArray(args, _) =>
        val dep = requestedType.asInstanceOf[TArray]
        val args2 = args.map(a => rebuildIR(ctx, a, env, memo))
        MakeArray.unify(ctx, args2, TArray(dep.elementType))
      case MakeStream(args, _, requiresMemoryManagementPerElement) =>
        val dep = requestedType.asInstanceOf[TStream]
        val args2 = args.map(a => rebuildIR(ctx, a, env, memo))
        MakeStream.unify(
          ctx,
          args2,
          requiresMemoryManagementPerElement,
          requestedType = TStream(dep.elementType),
        )
      case StreamZip(as, names, body, b, errorID) =>
        val (newAs, newNames) = as.lazyZip(names)
          .flatMap { case (a, name) =>
            if (memo.requestedType.contains(a)) Some((rebuildIR(ctx, a, env, memo), name)) else None
          }
          .unzip
        StreamZip(
          newAs,
          newNames,
          rebuildIR(
            ctx,
            body,
            env.bindEval(newNames.zip(newAs.map(a => TIterable.elementType(a.typ))): _*),
            memo,
          ),
          b,
          errorID,
        )
      case StreamMultiMerge(as, key) =>
        val eltType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val reqEltWithKey = unify(eltType, requestedEltType, selectKey(eltType, key))

        val newAs = as.map(a => rebuildIR(ctx, a, env, memo))
        val newAs2 = if (newAs.forall(_.typ == newAs(0).typ))
          newAs
        else
          newAs.map(a => upcast(ctx, a, TStream(reqEltWithKey)))

        StreamMultiMerge(newAs2, key)
      case MakeStruct(fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        // drop unnecessary field IRs
        val depFields = depStruct.fieldNames.toSet
        MakeStruct(fields.flatMap { case (f, fir) =>
          if (depFields.contains(f))
            Some(f -> rebuildIR(ctx, fir, env, memo))
          else {
            logger.info(s"Prune: MakeStruct: eliminating field '$f'")
            None
          }
        })
      case MakeTuple(fields) =>
        val depTuple = requestedType.asInstanceOf[TTuple]
        // drop unnecessary field IRs
        val depFieldIndices = depTuple.fieldIndex.keySet
        MakeTuple(fields.flatMap { case (i, f) =>
          if (depFieldIndices(i))
            Some(i -> rebuildIR(ctx, f, env, memo))
          else
            None
        })
      case InsertFields(old, fields, fieldOrder) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val depFields = depStruct.fieldNames.toSet
        val rebuiltChild = rebuildIR(ctx, old, env, memo)
        val preservedChildFields = rebuiltChild.typ.asInstanceOf[TStruct].fieldNames.toSet

        val insertOverwritesUnrequestedButPreservedField = fields.exists { case (fieldName, _) =>
          preservedChildFields.contains(fieldName) && !depFields.contains(fieldName)
        }

        val wrappedChild = if (insertOverwritesUnrequestedButPreservedField) {
          val selectedChildFields = preservedChildFields.filter(s => depFields.contains(s))
          SelectFields(
            rebuiltChild,
            rebuiltChild.typ.asInstanceOf[TStruct].fieldNames.filter(
              selectedChildFields.contains(_)
            ),
          )
        } else {
          rebuiltChild
        }

        InsertFields(
          wrappedChild,
          fields.flatMap { case (f, fir) =>
            if (depFields.contains(f))
              Some(f -> rebuildIR(ctx, fir, env, memo))
            else {
              logger.info(s"Prune: InsertFields: eliminating field '$f'")
              None
            }
          },
          fieldOrder.map(fds =>
            fds.filter(f =>
              depFields.contains(f) || wrappedChild.typ.asInstanceOf[TStruct].hasField(f)
            )
          ),
        )
      case SelectFields(old, fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val old2 = rebuildIR(ctx, old, env, memo)
        SelectFields(
          old2,
          fields.filter(f => old2.typ.asInstanceOf[TStruct].hasField(f) && depStruct.hasField(f)),
        )
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        if (!rStruct.hasField("rows"))
          if (rStruct.hasField("global"))
            MakeStruct(FastSeq("global" -> TableGetGlobals(rebuild(ctx, child, memo))))
          else
            MakeStruct(FastSeq())
        else {
          val rRowType = TIterable.elementType(rStruct.fieldType("rows")).asInstanceOf[TStruct]
          val rGlobType =
            rStruct.selfField("global").map(_.typ.asInstanceOf[TStruct]).getOrElse(TStruct())
          TableCollect(upcastTable(
            ctx,
            rebuild(ctx, child, memo),
            TableType(rowType = rRowType, FastSeq(), rGlobType),
            upcastRow = true,
            upcastGlobals = false,
          ))
        }
      case x: ApplyAggOp =>
        x.mapChildrenWithEnv(env) { (child, childEnv) =>
          rebuildIR(ctx, child.asInstanceOf[IR], childEnv, memo)
        }.asInstanceOf[ApplyAggOp]
      case x: ApplyScanOp =>
        x.mapChildrenWithEnv(env) { (child, childEnv) =>
          rebuildIR(ctx, child.asInstanceOf[IR], childEnv, memo)
        }.asInstanceOf[ApplyScanOp]
      case CollectDistributedArray(contexts, globals, cname, gname, body, dynamicID, staticID,
            tsd) =>
        val contexts2 = upcast(
          ctx,
          rebuildIR(ctx, contexts, env, memo),
          memo.requestedType.lookup(contexts).asInstanceOf[Type],
        )
        val globals2 = upcast(
          ctx,
          rebuildIR(ctx, globals, env, memo),
          memo.requestedType.lookup(globals).asInstanceOf[Type],
        )
        val body2 = rebuildIR(
          ctx,
          body,
          BindingEnv(Env(cname -> TIterable.elementType(contexts2.typ), gname -> globals2.typ)),
          memo,
        )
        val dynamicID2 = rebuildIR(ctx, dynamicID, env, memo)
        CollectDistributedArray(contexts2, globals2, cname, gname, body2, dynamicID2, staticID, tsd)
      case _ =>
        ir.mapChildrenWithEnv(env) { (child, childEnv) =>
          child match {
            case valueIR: IR => rebuildIR(ctx, valueIR, childEnv, memo)
            case mir: MatrixIR => rebuild(ctx, mir, memo)
            case tir: TableIR => rebuild(ctx, tir, memo)
            case bmir: BlockMatrixIR =>
              bmir // NOTE Currently no BlockMatrixIRs would have dead fields
          }
        }.asInstanceOf[IR]
    }
  }

  def upcast(ctx: ExecuteContext, ir: IR, rType: Type): IR = {
    if (ir.typ == rType)
      ir
    else {
      val result = ir.typ match {
        case tstruct: TStruct =>
          if (rType.asInstanceOf[TStruct].fields.forall(f => tstruct.field(f.name).typ == f.typ)) {
            SelectFields(ir, rType.asInstanceOf[TStruct].fields.map(f => f.name))
          } else {
            bindIR(ir) { ref =>
              val ms = MakeStruct(rType.asInstanceOf[TStruct].fields.map { f =>
                f.name -> upcast(ctx, GetField(ref, f.name), f.typ)
              })
              If(IsNA(ref), NA(ms.typ), ms)
            }
          }
        case ts: TStream =>
          val ra = rType.asInstanceOf[TStream]
          val uid = freshName()
          val ref = Ref(uid, ts.elementType)
          StreamMap(ir, uid, upcast(ctx, ref, ra.elementType))
        case ts: TArray =>
          val ra = rType.asInstanceOf[TArray]
          val uid = freshName()
          val ref = Ref(uid, ts.elementType)
          ToArray(StreamMap(ToStream(ir), uid, upcast(ctx, ref, ra.elementType)))
        case _: TTuple =>
          bindIR(ir) { ref =>
            val mt = MakeTuple(rType.asInstanceOf[TTuple]._types.map { tupleField =>
              tupleField.index -> upcast(
                ctx,
                GetTupleElement(ref, tupleField.index),
                tupleField.typ,
              )
            })
            If(IsNA(ref), NA(mt.typ), mt)
          }
        case _: TDict =>
          val rd = rType.asInstanceOf[TDict]
          ToDict(upcast(ctx, ToStream(ir), TArray(rd.elementType)))
        case _: TSet =>
          val rs = rType.asInstanceOf[TSet]
          ToSet(upcast(ctx, ToStream(ir), TSet(rs.elementType)))
        case _ => ir
      }

      assert(result.typ == rType, s"${Pretty(ctx, result)}, ${result.typ}, $rType")
      result
    }
  }

  def upcast(
    ctx: ExecuteContext,
    ir: MatrixIR,
    rType: MatrixType,
    upcastRows: Boolean = true,
    upcastCols: Boolean = true,
    upcastGlobals: Boolean = true,
    upcastEntries: Boolean = true,
  ): MatrixIR = {
    if (ir.typ == rType || !(upcastRows || upcastCols || upcastGlobals || upcastEntries))
      ir
    else {
      var mt = ir

      if (upcastRows && (ir.typ.rowKey != rType.rowKey)) {
        assert(ir.typ.rowKey.startsWith(rType.rowKey))
        mt = MatrixKeyRowsBy(mt, rType.rowKey)
      }

      if (upcastEntries && mt.typ.entryType != rType.entryType)
        mt = MatrixMapEntries(
          mt,
          upcast(ctx, Ref(MatrixIR.entryName, mt.typ.entryType), rType.entryType),
        )

      if (upcastRows && mt.typ.rowType != rType.rowType)
        mt = MatrixMapRows(mt, upcast(ctx, Ref(MatrixIR.rowName, mt.typ.rowType), rType.rowType))

      if (upcastCols && (mt.typ.colType != rType.colType || mt.typ.colKey != rType.colKey)) {
        mt = MatrixMapCols(
          mt,
          upcast(ctx, Ref(MatrixIR.colName, mt.typ.colType), rType.colType),
          if (rType.colKey == mt.typ.colKey) None else Some(rType.colKey),
        )
      }

      if (upcastGlobals && mt.typ.globalType != rType.globalType)
        mt = MatrixMapGlobals(
          mt,
          upcast(ctx, Ref(MatrixIR.globalName, ir.typ.globalType), rType.globalType),
        )

      mt
    }
  }

  def upcastTable(
    ctx: ExecuteContext,
    ir: TableIR,
    rType: TableType,
    upcastRow: Boolean = true,
    upcastGlobals: Boolean = true,
  ): TableIR = {
    if (ir.typ == rType)
      ir
    else {
      var table = ir
      if (ir.typ.key != rType.key) {
        assert(ir.typ.key.startsWith(rType.key))
        table = TableKeyBy(table, rType.key)
      }
      if (upcastRow && ir.typ.rowType != rType.rowType) {
        table =
          TableMapRows(table, upcast(ctx, Ref(TableIR.rowName, table.typ.rowType), rType.rowType))
      }
      if (upcastGlobals && ir.typ.globalType != rType.globalType) {
        table =
          TableMapGlobals(
            table,
            upcast(ctx, Ref(TableIR.globalName, table.typ.globalType), rType.globalType),
          )
      }
      table
    }
  }
}
