package is.hail.expr.ir

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable


object PruneDeadFields {

  case class ComputeMutableState(requestedType: Memo[BaseType], relationalRefs: mutable.HashMap[String, BoxedArrayBuilder[Type]]) {
    def rebuildState: RebuildMutableState = RebuildMutableState(requestedType, mutable.HashMap.empty)
  }

  case class RebuildMutableState(requestedType: Memo[BaseType], relationalRefs: mutable.HashMap[String, Type])

  def subsetType(t: Type, path: Array[String], index: Int = 0): Type = {
    if (index == path.length)
      PruneDeadFields.minimal(t)
    else
      t match {
        case ts: TStruct => TStruct(path(index) -> subsetType(ts.field(path(index)).typ, path, index + 1))
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
        case (TNDArray(et1, ndims1), TNDArray(et2, ndims2)) => (ndims1 == ndims2) && isSupertype(et1, et2)
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

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    try {
      val irCopy = ir.deepCopy()
      val ms = ComputeMutableState(Memo.empty[BaseType], mutable.HashMap.empty)
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
          rebuildIR(ctx, vir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)), ms.rebuildState)
      }
    } catch {
      case e: Throwable => fatal(s"error trying to rebuild IR:\n${ Pretty(ctx, ir, elideLiterals = true) }", e)
    }
  }

  def selectKey(t: TStruct, k: IndexedSeq[String]): TStruct = t.filterSet(k.toSet)._1

  def minimal(tt: TableType): TableType = {
    TableType(
      rowType = TStruct.empty,
      key = FastSeq(),
      globalType = TStruct.empty
    )
  }

  def minimal(mt: MatrixType): MatrixType = {
    MatrixType(
      rowKey = FastSeq(),
      colKey = FastSeq(),
      rowType = TStruct.empty,
      colType = TStruct.empty,
      globalType = TStruct.empty,
      entryType = TStruct.empty
    )
  }

  def minimal[T <: Type](base: T): T = {
    val result = base match {
      case ts: TStruct => TStruct.empty
      case ta: TArray => TArray(minimal(ta.elementType))
      case ta: TStream => TStream(minimal(ta.elementType))
      case t => t
    }
    result.asInstanceOf[T]
  }

  def minimalBT[T <: BaseType](base: T): T = {
    (base match {
      case tt: TableType => minimal(tt)
      case mt: MatrixType => minimal(mt)
      case t: Type => minimal(t)
    }).asInstanceOf[T]
  }

  def unifyKey(children: Seq[IndexedSeq[String]]): IndexedSeq[String] = {
    children.foldLeft(FastSeq[String]()) { case (comb, k) => if (k.length > comb.length) k else comb }
  }

  def unifyBaseType(base: BaseType, children: BaseType*): BaseType = unifyBaseTypeSeq(base, children)

  def unifyBaseTypeSeq(base: BaseType, _children: Seq[BaseType]): BaseType = {
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
            globalType = unify(tt.globalType, ttChildren.map(_.globalType): _*)
          )
        case mt: MatrixType =>
          val mtChildren = children.map(_.asInstanceOf[MatrixType])
          mt.copy(
            rowKey = unifyKey(mtChildren.map(_.rowKey)),
            colKey = unifyKey(mtChildren.map(_.colKey)),
            globalType = unifySeq(mt.globalType, mtChildren.map(_.globalType)),
            rowType = unifySeq(mt.rowType, mtChildren.map(_.rowType)),
            entryType = unifySeq(mt.entryType, mtChildren.map(_.entryType)),
            colType = unifySeq(mt.colType, mtChildren.map(_.colType))
          )
        case t: Type =>
          if (children.isEmpty)
            return minimal(t)
          t match {
            case ts: TStruct =>
              val subStructs = children.map(_.asInstanceOf[TStruct])
              val fieldArrays = Array.fill(ts.fields.length)(new BoxedArrayBuilder[Type])

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
                val ab = fieldArrays(oldIdx)
                if (ab.nonEmpty) {
                  val oldField = ts.fields(oldIdx)
                  subFields(newIdx) = Field(oldField.name, unifySeq(oldField.typ, ab.result()), newIdx)
                  newIdx += 1
                }
                oldIdx += 1
              }
              TStruct(subFields)
            case tt: TTuple =>
              val subTuples = children.map(_.asInstanceOf[TTuple])

              val fieldArrays = Array.fill(tt.size)(new BoxedArrayBuilder[Type])

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
                val ab = fieldArrays(oldIdx)
                if (ab.nonEmpty) {
                  val oldField = tt._types(oldIdx)
                  subFields(newIdx) = TupleField(oldField.index, unifySeq(oldField.typ, ab.result()))
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
                throw new RuntimeException(s"invalid unification:\n  base:  ${ t.parsableString() }${ badChildren.mkString("\n") }")
              }
              base
          }
      }
    } catch {
      case e: RuntimeException =>
        throw new RuntimeException(s"failed to unify children while unifying:\n  base:  ${ base }\n${ _children.mkString("\n") }", e)
    }
  }

  def unify[T <: BaseType](base: T, children: T*): T = unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def unifySeq[T <: BaseType](base: T, children: Seq[T]): T = unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def unifyEnvs(envs: BindingEnv[BoxedArrayBuilder[Type]]*): BindingEnv[BoxedArrayBuilder[Type]] = unifyEnvsSeq(envs)

  def concatEnvs(envs: Seq[Env[BoxedArrayBuilder[Type]]]): Env[BoxedArrayBuilder[Type]] = {
    val lc = envs.lengthCompare(1)
    if (lc < 0)
      Env.empty
    else {
      var e1 = envs.head
      envs.iterator
        .drop(1)
        .foreach { e =>
          e.m.foreach { case (k, v) =>
            e1.lookupOption(k) match {
              case Some(ab) => ab ++= v.result()
              case None => e1 = e1.bind(k, v.clone())
            }
          }
        }

      e1
    }
  }

  def unifyEnvsSeq(envs: Seq[BindingEnv[BoxedArrayBuilder[Type]]]): BindingEnv[BoxedArrayBuilder[Type]] = {
    val lc = envs.lengthCompare(1)
    if (lc < 0)
      BindingEnv.empty[BoxedArrayBuilder[Type]]
    else if (lc == 0)
      envs.head
    else {
      val evalEnv = concatEnvs(envs.map(_.eval))
      val aggEnv = if (envs.exists(_.agg.isDefined)) Some(concatEnvs(envs.flatMap(_.agg))) else None
      val scanEnv = if (envs.exists(_.scan.isDefined)) Some(concatEnvs(envs.flatMap(_.scan))) else None
      BindingEnv(evalEnv, aggEnv, scanEnv)
    }
  }

  def relationalTypeToEnv(bt: BaseType): BindingEnv[Type] = {
    val e = bt match {
      case tt: TableType =>
        Env.empty[Type]
          .bind("row", tt.rowType)
          .bind("global", tt.globalType)
      case mt: MatrixType =>
        Env.empty[Type]
          .bind("global", mt.globalType)
          .bind("sa", mt.colType)
          .bind("va", mt.rowType)
          .bind("g", mt.entryType)
    }
    BindingEnv(e, Some(e), Some(e))
  }

  def uses(name: String, env: Env[BoxedArrayBuilder[Type]]): Array[Type] =
    env.lookupOption(name).map(_.result()).getOrElse(Array.empty)

  def memoizeTableIR(
    ctx: ExecuteContext,
    tir: TableIR,
    requestedType: TableType,
    memo: ComputeMutableState
  ) {
    memo.requestedType.bind(tir, requestedType)
    tir match {
      case TableRead(_, _, _) =>
      case TableLiteral(_, _, _, _) =>
      case TableParallelize(rowsAndGlobal, _) =>
        memoizeValueIR(ctx, rowsAndGlobal, TStruct("rows" -> TArray(requestedType.rowType), "global" -> requestedType.globalType), memo)
      case TableRange(_, _) =>
      case TableRepartition(child, _, _) => memoizeTableIR(ctx, child, requestedType, memo)
      case TableHead(child, _) => memoizeTableIR(ctx, child, TableType(
        key = child.typ.key,
        rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, child.typ.key), requestedType.rowType),
        globalType = requestedType.globalType), memo)
      case TableTail(child, _) => memoizeTableIR(ctx, child, TableType(
        key = child.typ.key,
        rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, child.typ.key), requestedType.rowType),
        globalType = requestedType.globalType), memo)

      case TableGen(contexts, globals, cname, gname, body, _, _) =>
        val bodyEnv = memoizeValueIR(ctx, body, TStream(requestedType.rowType), memo)
        // Contexts are only used in the body so we only need to keep the fields used therein
        val contextsElemType = unifySeq(TIterable.elementType(contexts.typ), uses(cname, bodyEnv.eval))
        // Globals are exported and used in body, so keep the union of the used fields
        val globalsType = unifySeq(globals.typ, uses(gname, bodyEnv.eval) :+ requestedType.globalType)
        memoizeValueIR(ctx, contexts, TStream(contextsElemType), memo)
        memoizeValueIR(ctx, globals, globalsType, memo)

      case TableJoin(left, right, _, joinKey) =>
        val lk = unifyKey(FastSeq(requestedType.key.take(left.typ.key.length), left.typ.key.take(joinKey)))
        val lkSet = lk.toSet
        val leftDep = TableType(
          key = lk,
          rowType = TStruct(left.typ.rowType.fieldNames.flatMap(f =>
            if (lkSet.contains(f))
              Some(f -> left.typ.rowType.field(f).typ)
            else
              requestedType.rowType.fieldOption(f).map(reqF => f -> reqF.typ)): _*),
          globalType = TStruct(left.typ.globalType.fieldNames.flatMap(f =>
            requestedType.globalType.fieldOption(f).map(reqF => f -> reqF.typ)): _*))
        memoizeTableIR(ctx, left, leftDep, memo)

        val rk = right.typ.key.take(joinKey + math.max(0, requestedType.key.length - left.typ.key.length))
        val rightKeyFields = rk.toSet
        val rightDep = TableType(
          key = rk,
          rowType = TStruct(right.typ.rowType.fieldNames.flatMap(f =>
            if (rightKeyFields.contains(f))
              Some(f -> right.typ.rowType.field(f).typ)
            else
              requestedType.rowType.fieldOption(f).map(reqF => f -> reqF.typ)): _*),
          globalType = TStruct(right.typ.globalType.fieldNames.flatMap(f =>
            requestedType.globalType.fieldOption(f).map(reqF => f -> reqF.typ)): _*))
        memoizeTableIR(ctx, right, rightDep, memo)
      case TableLeftJoinRightDistinct(left, right, root) =>
        val fieldDep = requestedType.rowType.fieldOption(root).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val rightDep = TableType(
              key = right.typ.key,
              rowType = unify(
                right.typ.rowType,
                FastSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                  FastSeq(struct): _*),
              globalType = minimal(right.typ.globalType))
            memoizeTableIR(ctx, right, rightDep, memo)

            val lk = unifyKey(FastSeq(left.typ.key.take(right.typ.key.length), requestedType.key))
            val leftDep = TableType(
              key = lk,
              rowType = unify(left.typ.rowType, requestedType.rowType.filterSet(Set(root), include = false)._1,
                selectKey(left.typ.rowType, lk)),
              globalType = requestedType.globalType)
            memoizeTableIR(ctx, left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(ctx, left, requestedType, memo)
        }
      case TableIntervalJoin(left, right, root, product) =>
        val fieldDep = requestedType.rowType.fieldOption(root).map { field =>
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
                  FastSeq(struct): _*),
              globalType = minimal(right.typ.globalType))
            memoizeTableIR(ctx, right, rightDep, memo)

            val lk = unifyKey(FastSeq(left.typ.key.take(right.typ.key.length), requestedType.key))
            val leftDep = TableType(
              key = lk,
              rowType = unify(left.typ.rowType, requestedType.rowType.filterSet(Set(root), include = false)._1,
                selectKey(left.typ.rowType, lk)),
              globalType = requestedType.globalType)
            memoizeTableIR(ctx, left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(ctx, left, requestedType, memo)
        }
      case TableMultiWayZipJoin(children, fieldName, globalName) =>
        val gType = requestedType.globalType.fieldOption(globalName)
          .map(_.typ.asInstanceOf[TArray].elementType)
          .getOrElse(TStruct.empty).asInstanceOf[TStruct]
        val rType = requestedType.rowType.fieldOption(fieldName)
          .map(_.typ.asInstanceOf[TArray].elementType)
          .getOrElse(TStruct.empty).asInstanceOf[TStruct]
        val child1 = children.head
        val dep = TableType(
          key = child1.typ.key,
          rowType = TStruct(child1.typ.rowType.fieldNames.flatMap(f =>
            child1.typ.keyType.fieldOption(f).orElse(rType.fieldOption(f)).map(reqF => f -> reqF.typ)
          ): _*),
          globalType = gType)
        children.foreach(memoizeTableIR(ctx, _, dep, memo))
      case TableExplode(child, path) =>
        def getExplodedField(typ: TableType): Type = typ.rowType.queryTyped(path.toList)._1

        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExlosionFieldType = try {
          val t = getExplodedField(requestedType)
          preExplosionFieldType match {
            case ta: TArray => TArray(t)
            case ts: TSet => ts.copy(elementType = t)
          }
        } catch {
          case e: AnnotationPathException => minimal(preExplosionFieldType)
        }
        val dep = requestedType.copy(rowType = unify(child.typ.rowType,
          requestedType.rowType.insert(prunedPreExlosionFieldType, path.toList)._1.asInstanceOf[TStruct]))
        memoizeTableIR(ctx, child, dep, memo)
      case TableFilter(child, pred) =>
        val irDep = memoizeAndGetDep(ctx, pred, pred.typ, child.typ, memo)
        memoizeTableIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case TableKeyBy(child, _, isSorted) =>
        val reqKey = requestedType.key
        val isPrefix = reqKey.zip(child.typ.key).forall { case (l, r) => l == r }
        val childReqKey = if (isSorted)
          child.typ.key
        else if (isPrefix)
          if  (reqKey.length <= child.typ.key.length) reqKey else child.typ.key
        else FastSeq()

        memoizeTableIR(ctx, child, TableType(
          key = childReqKey,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, childReqKey), requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableOrderBy(child, sortFields) =>
        val k = if (sortFields.forall(_.sortOrder == Ascending) && child.typ.key.startsWith(sortFields.map(_.field)))
          child.typ.key
        else
          FastSeq()
        memoizeTableIR(ctx, child, TableType(
          key = k,
          rowType = unify(child.typ.rowType,
            selectKey(child.typ.rowType, sortFields.map(_.field) ++ k),
            requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableDistinct(child) =>
        val dep = TableType(key = child.typ.key,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.key)),
          globalType = requestedType.globalType)
        memoizeTableIR(ctx, child, dep, memo)
      case TableMapPartitions(child, gName, pName, body, requestedKey, _) =>
        val requestedKeyStruct = child.typ.keyType.truncate(math.max(requestedType.key.length, requestedKey))
        val reqRowsType = unify(body.typ, TStream(requestedType.rowType), TStream(requestedKeyStruct))
        val bodyDep = memoizeValueIR(ctx, body, reqRowsType, memo)
        val depGlobalType = unifySeq(
          child.typ.globalType,
          uses(gName, bodyDep.eval) :+ requestedType.globalType
        )
        val depRowType = unifySeq(
          child.typ.rowType,
          uses(pName, bodyDep.eval).map(TIterable.elementType) :+ requestedKeyStruct)
        val dep = TableType(
          key = requestedKeyStruct.fieldNames,
          rowType = depRowType.asInstanceOf[TStruct],
          globalType = depGlobalType.asInstanceOf[TStruct])
        memoizeTableIR(ctx, child, dep, memo)
      case TableMapRows(child, newRow) =>
        val (reqKey, reqRowType) = if (ContainsScan(newRow))
          (child.typ.key, unify(newRow.typ, requestedType.rowType, selectKey(newRow.typ.asInstanceOf[TStruct], child.typ.key)))
        else
          (requestedType.key, requestedType.rowType)
        val rowDep = memoizeAndGetDep(ctx, newRow, reqRowType, child.typ, memo)

        val dep = TableType(
          key = reqKey,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, reqKey), rowDep.rowType),
          globalType = unify(child.typ.globalType, requestedType.globalType, rowDep.globalType)
        )
        memoizeTableIR(ctx, child, dep, memo)
      case TableMapGlobals(child, newGlobals) =>
        val globalDep = memoizeAndGetDep(ctx, newGlobals, requestedType.globalType, child.typ, memo)
        memoizeTableIR(ctx, child, unify(child.typ, requestedType.copy(globalType = globalDep.globalType), globalDep), memo)
      case TableAggregateByKey(child, expr) =>
        val exprRequestedType = requestedType.rowType.filter(f => expr.typ.asInstanceOf[TStruct].hasField(f.name))._1
        val aggDep = memoizeAndGetDep(ctx, expr, exprRequestedType, child.typ, memo)
        memoizeTableIR(ctx, child, TableType(key = child.typ.key,
          rowType = unify(child.typ.rowType, aggDep.rowType, selectKey(child.typ.rowType, child.typ.key)),
          globalType = unify(child.typ.globalType, aggDep.globalType, requestedType.globalType)), memo)
      case TableKeyByAndAggregate(child, expr, newKey, _, _) =>
        val keyDep = memoizeAndGetDep(ctx, newKey, newKey.typ, child.typ, memo)
        val exprDep = memoizeAndGetDep(ctx, expr, requestedType.valueType, child.typ, memo)
        memoizeTableIR(ctx, child,
          TableType(
            key = FastSeq(), // note: this can deoptimize if prune runs before Simplify
            rowType = unify(child.typ.rowType, keyDep.rowType, exprDep.rowType),
            globalType = unify(child.typ.globalType, keyDep.globalType, exprDep.globalType, requestedType.globalType)),
          memo)
      case MatrixColsTable(child) =>
        val mtDep = minimal(child.typ).copy(
          globalType = requestedType.globalType,
          entryType = TStruct.empty,
          colType = requestedType.rowType,
          colKey = requestedType.key)
        memoizeMatrixIR(ctx, child, mtDep, memo)
      case MatrixRowsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = requestedType.globalType,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, requestedType.key), requestedType.rowType),
          rowKey = requestedType.key)
        memoizeMatrixIR(ctx, child, mtDep, memo)
      case MatrixEntriesTable(child) =>
        val mtDep = MatrixType(
          rowKey = requestedType.key.take(child.typ.rowKey.length),
          colKey = requestedType.key.drop(child.typ.rowKey.length),
          globalType = requestedType.globalType,
          colType = TStruct(
            child.typ.colType.fields.flatMap(f => requestedType.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)): _*),
          rowType = TStruct(
            child.typ.rowType.fields.flatMap(f => requestedType.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)): _*),
          entryType = TStruct(
            child.typ.entryType.fields.flatMap(f => requestedType.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)): _*)
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
            requestedType.globalType.field(colsFieldName).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
          else
            TStruct.empty,
          entryType = if (requestedType.rowType.hasField(entriesFieldName))
            requestedType.rowType.field(entriesFieldName).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
          else
            TStruct.empty,
          rowType = if (requestedType.rowType.hasField(entriesFieldName))
            requestedType.rowType.deleteKey(entriesFieldName)
          else
            requestedType.rowType)
        memoizeMatrixIR(ctx, child, childDep, memo)
      case TableRename(child, rowMap, globalMap) =>
        val rowMapRev = rowMap.map { case (k, v) => (v, k) }
        val globalMapRev = globalMap.map { case (k, v) => (v, k) }
        val childDep = TableType(
          rowType = requestedType.rowType.rename(rowMapRev),
          globalType = requestedType.globalType.rename(globalMapRev),
          key = requestedType.key.map(k => rowMapRev.getOrElse(k, k)))
        memoizeTableIR(ctx, child, childDep, memo)
      case TableFilterIntervals(child, _, _) =>
        memoizeTableIR(ctx, child, requestedType.copy(key = child.typ.key,
          rowType = PruneDeadFields.unify(child.typ.rowType,
            requestedType.rowType,
            PruneDeadFields.selectKey(child.typ.rowType, child.typ.key))), memo)
      case TableToTableApply(child, f) => memoizeTableIR(ctx, child, child.typ, memo)
      case MatrixToTableApply(child, _) => memoizeMatrixIR(ctx, child, child.typ, memo)
      case BlockMatrixToTableApply(bm, aux, _) =>
        memoizeBlockMatrixIR(ctx, bm, bm.typ, memo)
        memoizeValueIR(ctx, aux, aux.typ, memo)
      case BlockMatrixToTable(child) => memoizeBlockMatrixIR(ctx, child, child.typ, memo)
      case RelationalLetTable(name, value, body) =>
        memoizeTableIR(ctx, body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(ctx, value, unifySeq(value.typ, usages), memo)
    }
  }

  def memoizeMatrixIR(
    ctx: ExecuteContext,
    mir: MatrixIR,
    requestedType: MatrixType,
    memo: ComputeMutableState
  ) {
    memo.requestedType.bind(mir, requestedType)
    mir match {
      case MatrixFilterCols(child, pred) =>
        val irDep = memoizeAndGetDep(ctx, pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterRows(child, pred) =>
        val irDep = memoizeAndGetDep(ctx, pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterEntries(child, pred) =>
        val irDep = memoizeAndGetDep(ctx, pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType, irDep), memo)
      case MatrixUnionCols(left, right, joinType) =>
        val leftRequestedType = requestedType.copy(
          rowKey = left.typ.rowKey,
          rowType = unify(left.typ.rowType, requestedType.rowType, selectKey(left.typ.rowType, left.typ.rowKey))
        )
        val rightRequestedType = requestedType.copy(
          globalType = TStruct.empty,
          rowKey = right.typ.rowKey,
          rowType = unify(right.typ.rowType, requestedType.rowType, selectKey(right.typ.rowType, right.typ.rowKey)))
        memoizeMatrixIR(ctx, left, leftRequestedType, memo)
        memoizeMatrixIR(ctx, right, rightRequestedType, memo)
      case MatrixMapEntries(child, newEntries) =>
        val irDep = memoizeAndGetDep(ctx, newEntries, requestedType.entryType, child.typ, memo)
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

        memoizeMatrixIR(ctx, child, requestedType.copy(
          rowKey = childReqKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, childReqKey))),
          memo)
      case MatrixMapRows(child, newRow) =>
        val (reqKey, reqRowType) = if (ContainsScan(newRow))
          (child.typ.rowKey, unify(newRow.typ, requestedType.rowType, selectKey(newRow.typ.asInstanceOf[TStruct], child.typ.rowKey)))
        else
          (requestedType.rowKey, requestedType.rowType)

        val irDep = memoizeAndGetDep(ctx, newRow, reqRowType, child.typ, memo)
        val depMod = requestedType.copy(rowType = selectKey(child.typ.rowType, reqKey), rowKey = reqKey)
        memoizeMatrixIR(ctx, child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapCols(child, newCol, newKey) =>
        val irDep = memoizeAndGetDep(ctx, newCol, requestedType.colType, child.typ, memo)
        val reqKey =  newKey match {
          case Some(_) => FastSeq()
          case None => requestedType.colKey
        }
        val depMod = requestedType.copy(colType = selectKey(child.typ.colType, reqKey), colKey = reqKey)
        memoizeMatrixIR(ctx, child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapGlobals(child, newGlobals) =>
        val irDep = memoizeAndGetDep(ctx, newGlobals, requestedType.globalType, child.typ, memo)
        memoizeMatrixIR(ctx, child, unify(child.typ, requestedType.copy(globalType = irDep.globalType), irDep), memo)
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
              requestedColType.fieldOption(f.name)
                .map(requestedField => f.name -> requestedField.typ.asInstanceOf[TArray].elementType)
            }
          }: _*),
          rowType = requestedType.rowType,
          entryType = TStruct(requestedType.entryType.fields.map(f => f.copy(typ = f.typ.asInstanceOf[TArray].elementType))))
        memoizeMatrixIR(ctx, child, explodedDep, memo)
      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val irDepEntry = memoizeAndGetDep(ctx, entryExpr, requestedType.entryType, child.typ, memo)
        val irDepRow = memoizeAndGetDep(ctx, rowExpr, requestedType.rowValueStruct, child.typ, memo)
        val childDep = MatrixType(
          rowKey = child.typ.rowKey,
          colKey = requestedType.colKey,
          entryType = irDepEntry.entryType,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, child.typ.rowKey), irDepRow.rowType, irDepEntry.rowType),
          colType = unify(child.typ.colType, requestedType.colType, irDepEntry.colType, irDepRow.colType),
          globalType = unify(child.typ.globalType, requestedType.globalType, irDepEntry.globalType, irDepRow.globalType))
        memoizeMatrixIR(ctx, child, childDep, memo)
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val irDepEntry = memoizeAndGetDep(ctx, entryExpr, requestedType.entryType, child.typ, memo)
        val irDepCol = memoizeAndGetDep(ctx, colExpr, requestedType.colValueStruct, child.typ, memo)
        val childDep: MatrixType = MatrixType(
          rowKey = requestedType.rowKey,
          colKey = child.typ.colKey,
          colType = unify(child.typ.colType, irDepCol.colType, irDepEntry.colType, selectKey(child.typ.colType, child.typ.colKey)),
          globalType = unify(child.typ.globalType, requestedType.globalType, irDepEntry.globalType, irDepCol.globalType),
          rowType = unify(child.typ.rowType, irDepEntry.rowType, irDepCol.rowType, requestedType.rowType),
          entryType = irDepEntry.entryType)
        memoizeMatrixIR(ctx, child, childDep, memo)
      case MatrixAnnotateRowsTable(child, table, root, product) =>
        val fieldDep = requestedType.rowType.fieldOption(root).map { field =>
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
              globalType = minimal(table.typ.globalType))
            memoizeTableIR(ctx, table, tableDep, memo)

            val mk = unifyKey(FastSeq(child.typ.rowKey.take(tk.length), requestedType.rowKey))
            val matDep = requestedType.copy(
              rowKey = mk,
              rowType =
                unify(child.typ.rowType,
                  selectKey(child.typ.rowType, mk),
                  requestedType.rowType.filterSet(Set(root), include = false)._1))
            memoizeMatrixIR(ctx, child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(ctx, child, requestedType, memo)
        }
      case MatrixAnnotateColsTable(child, table, uid) =>
        val fieldDep = requestedType.colType.fieldOption(uid).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val tk = table.typ.key
            val tableDep = TableType(
              key = tk,
              rowType = unify(table.typ.rowType, struct, selectKey(table.typ.rowType, tk)),
              globalType = minimal(table.typ.globalType))
            memoizeTableIR(ctx, table, tableDep, memo)

            val mk = unifyKey(FastSeq(child.typ.colKey.take(table.typ.key.length), requestedType.colKey))
            val matDep = requestedType.copy(
              colKey = mk,
              colType = unify(child.typ.colType, requestedType.colType.filterSet(Set(uid), include = false)._1,
                selectKey(child.typ.colType, mk)))
            memoizeMatrixIR(ctx, child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(ctx, child, requestedType, memo)
        }
      case MatrixExplodeRows(child, path) =>
        def getExplodedField(typ: MatrixType): Type = typ.rowType.queryTyped(path.toList)._1

        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExlosionFieldType = try {
          val t = getExplodedField(requestedType)
          preExplosionFieldType match {
            case ta: TArray => TArray(t)
            case ts: TSet => ts.copy(elementType = t)
          }
        } catch {
          case e: AnnotationPathException => minimal(preExplosionFieldType)
        }
        val dep = requestedType.copy(rowType = unify(child.typ.rowType,
          requestedType.rowType.insert(prunedPreExlosionFieldType, path.toList)._1.asInstanceOf[TStruct]))
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixExplodeCols(child, path) =>
        def getExplodedField(typ: MatrixType): Type = typ.colType.queryTyped(path.toList)._1

        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExplosionFieldType = try {
          val t = getExplodedField(requestedType)
          preExplosionFieldType match {
            case ta: TArray => TArray(t)
            case ts: TSet => ts.copy(elementType = t)
          }
        } catch {
          case e: AnnotationPathException => minimal(preExplosionFieldType)
        }
        val dep = requestedType.copy(colType = unify(child.typ.colType,
          requestedType.colType.insert(prunedPreExplosionFieldType, path.toList)._1.asInstanceOf[TStruct]))
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixRepartition(child, _, _) =>
        memoizeMatrixIR(ctx, child, requestedType, memo)
      case MatrixUnionRows(children) =>
        memoizeMatrixIR(ctx, children.head, requestedType, memo)
        children.tail.foreach(memoizeMatrixIR(ctx, _, requestedType.copy(colType = requestedType.colKeyStruct), memo))
      case MatrixDistinctByRow(child) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.rowKey))
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixRowsHead(child, n) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.rowKey))
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixColsHead(child, n) => memoizeMatrixIR(ctx, child, requestedType, memo)
      case MatrixRowsTail(child, n) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.rowKey))
        )
        memoizeMatrixIR(ctx, child, dep, memo)
      case MatrixColsTail(child, n) => memoizeMatrixIR(ctx, child, requestedType, memo)
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        val m = Map(MatrixType.entriesIdentifier -> entriesFieldName)
        val childDep = child.typ.copy(
          key = requestedType.rowKey,
          globalType = unify(child.typ.globalType, requestedType.globalType, TStruct((colsFieldName, TArray(requestedType.colType)))),
          rowType = unify(child.typ.rowType, requestedType.rowType, TStruct((entriesFieldName, TArray(requestedType.entryType))))
        )
        memoizeTableIR(ctx, child, childDep, memo)
      case MatrixFilterIntervals(child, _, _) =>
        memoizeMatrixIR(ctx, child, requestedType.copy(rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType,
            requestedType.rowType,
            selectKey(child.typ.rowType, child.typ.rowKey))), memo)
      case MatrixToMatrixApply(child, f) => memoizeMatrixIR(ctx, child, child.typ, memo)
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
          entryType = requestedType.entryType.rename(entryMapRev))
        memoizeMatrixIR(ctx, child, childDep, memo)
      case RelationalLetMatrixTable(name, value, body) =>
        memoizeMatrixIR(ctx, body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(ctx, value, unifySeq(value.typ, usages), memo)
    }
  }

  def memoizeBlockMatrixIR(
    ctx: ExecuteContext,
    bmir: BlockMatrixIR,
    requestedType: BlockMatrixType,
    memo: ComputeMutableState
  ): Unit = {
    memo.requestedType.bind(bmir, requestedType)
    bmir match {
      case RelationalLetBlockMatrix(name, value, body) =>
        memoizeBlockMatrixIR(ctx, body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(ctx, value, unifySeq(value.typ, usages), memo)
      case _ =>
        bmir.children.foreach {
          case mir: MatrixIR => memoizeMatrixIR(ctx, mir, mir.typ, memo)
          case tir: TableIR => memoizeTableIR(ctx, tir, tir.typ, memo)
          case bmir: BlockMatrixIR => memoizeBlockMatrixIR(ctx, bmir, bmir.typ, memo)
          case ir: IR => memoizeValueIR(ctx, ir, ir.typ, memo)
        }
    }
  }

  def memoizeAndGetDep(
    ctx: ExecuteContext,
    ir: IR,
    requestedType: Type,
    base: TableType,
    memo: ComputeMutableState
  ): TableType = {
    val depEnv = memoizeValueIR(ctx, ir, requestedType, memo)
    val depEnvUnified = concatEnvs(FastSeq(depEnv.eval) ++ FastSeq(depEnv.agg, depEnv.scan).flatten)

    val expectedBindingSet = Set("row", "global")
    depEnvUnified.m.keys.foreach { k =>
      if (!expectedBindingSet.contains(k))
        throw new RuntimeException(s"found unexpected free variable in pruning: $k\n" +
          s"  ${ depEnv.pretty(_.result().mkString(",")) }\n" +
          s"  ${ Pretty(ctx, ir) }")
    }

    val min = minimal(base)
    val rowType = unifySeq(base.rowType,
      Array(min.rowType) ++ uses("row", depEnvUnified)
    )
    val globalType = unifySeq(base.globalType,
      Array(min.globalType) ++ uses("global", depEnvUnified)
    )
    TableType(key = FastSeq(),
      rowType = rowType.asInstanceOf[TStruct],
      globalType = globalType.asInstanceOf[TStruct])
  }

  def memoizeAndGetDep(
    ctx: ExecuteContext,
    ir: IR,
    requestedType: Type,
    base: MatrixType,
    memo: ComputeMutableState
  ): MatrixType = {
    val depEnv = memoizeValueIR(ctx, ir, requestedType, memo)
    val depEnvUnified = concatEnvs(FastSeq(depEnv.eval) ++ FastSeq(depEnv.agg, depEnv.scan).flatten)

    val expectedBindingSet = Set("va", "sa", "g", "global", "n_rows", "n_cols")
    depEnvUnified.m.keys.foreach { k =>
      if (!expectedBindingSet.contains(k))
        throw new RuntimeException(s"found unexpected free variable in pruning: $k\n  ${ Pretty(ctx, ir) }")
    }

    val min = minimal(base)
    val globalType = unifySeq(base.globalType,
      Array(min.globalType) ++ uses("global", depEnvUnified))
      .asInstanceOf[TStruct]
    val rowType = unifySeq(base.rowType,
      Array(min.rowType) ++ uses("va", depEnvUnified))
      .asInstanceOf[TStruct]
    val colType = unifySeq(base.colType,
      Array(min.colType) ++ uses("sa", depEnvUnified))
      .asInstanceOf[TStruct]
    val entryType = unifySeq(base.entryType,
      Array(min.entryType) ++ uses("g", depEnvUnified))
      .asInstanceOf[TStruct]

    if (rowType.hasField(MatrixType.entriesIdentifier))
      throw new RuntimeException(s"prune: found dependence on entry array in row binding:\n${ Pretty(ctx, ir) }")

    MatrixType(
      rowKey = FastSeq(),
      colKey = FastSeq(),
      globalType = globalType,
      colType = colType,
      rowType = rowType,
      entryType = entryType)
  }

  /**
    * This function does *not* necessarily bind each child node in `memo`.
    * Known dead code is not memoized. For instance:
    *
    *   ir = MakeStruct(Seq("a" -> (child1), "b" -> (child2)))
    *   requestedType = TStruct("a" -> (reqType of a))
    *
    * In the above, `child2` will not be memoized because `ir` does not require
    * any of the "b" dependencies in order to create its own requested type,
    * which only contains "a".
    */
  def memoizeValueIR(
    ctx: ExecuteContext,
    ir: IR,
    requestedType: Type,
    memo: ComputeMutableState
  ): BindingEnv[BoxedArrayBuilder[Type]] = {
    memo.requestedType.bind(ir, requestedType)
    ir match {
      case IsNA(value) => memoizeValueIR(ctx, value, minimal(value.typ), memo)
      case CastRename(v, _typ) =>
        def recur(reqType: Type, castType: Type, baseType: Type): Type = {
          ((reqType, castType, baseType): @unchecked) match {
            case (TStruct(reqFields), cast: TStruct, base: TStruct) =>
              TStruct(reqFields.map { f =>
                val idx = cast.fieldIdx(f.name)
                Field(base.fieldNames(idx), recur(f.typ, cast.types(idx), base.types(idx)), f.index)
              })
            case (TTuple(req), TTuple(cast), TTuple(base)) =>
              assert(base.length == cast.length)
              val castFields = cast.map { f => f.index -> f.typ }.toMap
              val baseFields = base.map { f => f.index -> f.typ }.toMap
              TTuple(req.map { f => TupleField(f.index, recur(f.typ, castFields(f.index), baseFields(f.index)))})
            case (TArray(req), TArray(cast), TArray(base)) =>
              TArray(recur(req, cast, base))
            case (TSet(req), TSet(cast), TSet(base)) =>
              TSet(recur(req, cast, base))
            case (TDict(reqK, reqV), TDict(castK, castV), TDict(baseK, baseV)) =>
              TDict(recur(reqK, castK, baseK), recur(reqV, castV, baseV))
            case (TInterval(req), TInterval(cast), TInterval(base)) =>
              TInterval(recur(req, cast, base))
            case _ => reqType
          }
        }

        memoizeValueIR(ctx, v, recur(requestedType, _typ, v.typ), memo)
      case If(cond, cnsq, alt) =>
        unifyEnvs(
          memoizeValueIR(ctx, cond, cond.typ, memo),
          memoizeValueIR(ctx, cnsq, requestedType, memo),
          memoizeValueIR(ctx, alt, requestedType, memo)
        )
      case Coalesce(values) => unifyEnvsSeq(values.map(memoizeValueIR(ctx, _, requestedType, memo)))
      case Consume(value) => memoizeValueIR(ctx, value, value.typ, memo)
      case Let(name, value, body) =>
        val bodyEnv = memoizeValueIR(ctx, body, requestedType, memo)
        val valueType = unifySeq(value.typ, uses(name, bodyEnv.eval))
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(ctx, value, valueType, memo)
        )
      case AggLet(name, value, body, isScan) =>
        val bodyEnv = memoizeValueIR(ctx, body, requestedType, memo)
        if (isScan) {
          val valueType = unifySeq(value.typ, uses(name, bodyEnv.scanOrEmpty))
          val valueEnv = memoizeValueIR(ctx, value, valueType, memo)
          unifyEnvs(
            bodyEnv.copy(scan = bodyEnv.scan.map(_.delete(name))),
            valueEnv.copy(eval = Env.empty, scan = Some(valueEnv.eval))
          )
        } else {
          val valueType = unifySeq(value.typ, uses(name, bodyEnv.aggOrEmpty))
          val valueEnv = memoizeValueIR(ctx, value, valueType, memo)
          unifyEnvs(
            bodyEnv.copy(agg = bodyEnv.agg.map(_.delete(name))),
            valueEnv.copy(eval = Env.empty, agg = Some(valueEnv.eval))
          )
        }
      case Ref(name, t) =>
        val ab = new BoxedArrayBuilder[Type]()
        ab += requestedType
        BindingEnv.empty.bindEval(name -> ab)
      case RelationalLet(name, value, body) =>
        val e = memoizeValueIR(ctx, body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(ctx, value, unifySeq(value.typ, usages), memo)
        e
      case RelationalRef(name, _) =>
        memo.relationalRefs.getOrElseUpdate(name, new BoxedArrayBuilder[Type]) += requestedType
        BindingEnv.empty
      case MakeArray(args, _) =>
        val eltType = TIterable.elementType(requestedType)
        unifyEnvsSeq(args.map(a => memoizeValueIR(ctx, a, eltType, memo)))
      case MakeStream(args, _, _) =>
        val eltType = TIterable.elementType(requestedType)
        unifyEnvsSeq(args.map(a => memoizeValueIR(ctx, a, eltType, memo)))
      case ArrayRef(a, i, s) =>
        unifyEnvs(
          memoizeValueIR(ctx, a, TArray(requestedType), memo),
          memoizeValueIR(ctx, i, i.typ, memo),
          memoizeValueIR(ctx, s, s.typ, memo)
        )
      case ArrayLen(a) =>
        memoizeValueIR(ctx, a, minimal(a.typ), memo)
      case StreamTake(a, len) =>
        unifyEnvs(
          memoizeValueIR(ctx, a, requestedType, memo),
          memoizeValueIR(ctx, len, len.typ, memo))
      case StreamDrop(a, len) =>
        unifyEnvs(
          memoizeValueIR(ctx, a, requestedType, memo),
          memoizeValueIR(ctx, len, len.typ, memo))
      case StreamWhiten(a, newChunk, prevWindow, _, _, _, _, _) =>
        val matType = TNDArray(TFloat64, Nat(2))
        val unifiedStructType = unify(
          a.typ.asInstanceOf[TStream].elementType,
          requestedType.asInstanceOf[TStream].elementType,
          TStruct((newChunk, matType), (prevWindow, matType)))
        unifyEnvs(
          memoizeValueIR(ctx, a, TStream(unifiedStructType), memo))
      case StreamMap(a, name, body) =>
        val bodyEnv = memoizeValueIR(ctx, body,
          TIterable.elementType(requestedType),
          memo
        )
        val valueType = unifySeq(TIterable.elementType(a.typ), uses(name, bodyEnv.eval))
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case StreamGrouped(a, size) =>
        unifyEnvs(
          memoizeValueIR(ctx, a, TIterable.elementType(requestedType), memo),
          memoizeValueIR(ctx, size, size.typ, memo))
      case StreamGroupByKey(a, key, _) =>
        val reqStructT = tcoerce[TStruct](tcoerce[TStream](tcoerce[TStream](requestedType).elementType).elementType)
        val origStructT = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
        memoizeValueIR(ctx, a, TStream(unify(origStructT, reqStructT, selectKey(origStructT, key))), memo)
      case StreamZip(as, names, body, behavior, _) =>
        val bodyEnv = memoizeValueIR(ctx, body,
          TIterable.elementType(requestedType),
          memo)
        val valueTypes = (names, as).zipped.map { (name, a) =>
          bodyEnv.eval.lookupOption(name).map(ab => unifySeq(tcoerce[TStream](a.typ).elementType, ab.result()))
        }
        if (behavior == ArrayZipBehavior.AssumeSameLength && valueTypes.forall(_.isEmpty)) {
          unifyEnvs(memoizeValueIR(ctx, as.head, TStream(minimal(tcoerce[TStream](as.head.typ).elementType)), memo) +:
                      Array(bodyEnv.deleteEval(names)): _*)
        } else {
          unifyEnvs(
            (as, valueTypes).zipped.map { (a, vtOption) =>
              val at = tcoerce[TStream](a.typ)
              if (behavior == ArrayZipBehavior.AssumeSameLength) {
                vtOption.map { vt =>
                  memoizeValueIR(ctx, a, TStream(vt), memo)
                }.getOrElse(BindingEnv.empty)
              } else
                memoizeValueIR(ctx, a, TStream(vtOption.getOrElse(minimal(at.elementType))), memo)
            } ++ Array(bodyEnv.deleteEval(names)): _*)
        }
      case StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val eltType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val bodyEnv = memoizeValueIR(ctx, joinF, requestedEltType, memo)
        val childRequestedEltType = unifySeq(
          eltType,
          uses(curVals, bodyEnv.eval).map(TIterable.elementType) :+ selectKey(eltType, key)
        )
        unifyEnvsSeq(as.map(memoizeValueIR(ctx, _, TStream(childRequestedEltType), memo)))
      case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, joinF) =>
        val baseEltType = tcoerce[TStruct](TIterable.elementType(makeProducer.typ))
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val bodyEnv = memoizeValueIR(ctx, joinF, requestedEltType, memo)
        val producerRequestedEltType = unifySeq(
          baseEltType,
          uses(curVals, bodyEnv.eval).map(TIterable.elementType) :+ selectKey(baseEltType, key)
        )
        val producerEnv = memoizeValueIR(ctx, makeProducer, TStream(producerRequestedEltType), memo)
        val ctxEnv = memoizeValueIR(ctx, contexts, TArray(unifySeq(TIterable.elementType(contexts.typ), uses(ctxName, producerEnv.eval))), memo)
        unifyEnvsSeq(Array(bodyEnv, producerEnv, ctxEnv))
      case StreamMultiMerge(as, key) =>
        val eltType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
        val requestedEltType = tcoerce[TStream](requestedType).elementType
        val childRequestedEltType = unify(eltType, requestedEltType, selectKey(eltType, key))
        unifyEnvsSeq(as.map(memoizeValueIR(ctx, _, TStream(childRequestedEltType), memo)))
      case StreamFilter(a, name, cond) =>
        val bodyEnv = memoizeValueIR(ctx, cond, cond.typ, memo)
        val valueType = unifySeq(
          TIterable.elementType(a.typ),
          FastSeq(TIterable.elementType(requestedType)) ++ uses(name, bodyEnv.eval)
        )
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case StreamTakeWhile(a, name, cond) =>
        val bodyEnv = memoizeValueIR(ctx, cond, cond.typ, memo)
        val valueType = unifySeq(
          TIterable.elementType(a.typ),
          FastSeq(TIterable.elementType(requestedType)) ++ uses(name, bodyEnv.eval)
        )
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case StreamDropWhile(a, name, cond) =>
        val bodyEnv = memoizeValueIR(ctx, cond, cond.typ, memo)
        val valueType = unifySeq(
          TIterable.elementType(a.typ),
          FastSeq(TIterable.elementType(requestedType)) ++ uses(name, bodyEnv.eval)
        )
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case StreamFlatMap(a, name, body) =>
        val bodyEnv = memoizeValueIR(ctx, body, requestedType, memo)
        val valueType = unifySeq(TIterable.elementType(a.typ), uses(name, bodyEnv.eval))
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case StreamFold(a, zero, accumName, valueName, body) =>
        val zeroEnv = memoizeValueIR(ctx, zero, zero.typ, memo)
        val bodyEnv = memoizeValueIR(ctx, body, body.typ, memo)
        val valueType = unifySeq(TIterable.elementType(a.typ), uses(valueName, bodyEnv.eval))

        unifyEnvs(
          zeroEnv,
          bodyEnv.deleteEval(valueName).deleteEval(accumName),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case StreamFold2(a, accum, valueName, seq, res) =>
        val zeroEnvs = accum.map { case (name, zval) => memoizeValueIR(ctx, zval, zval.typ, memo) }
        val seqEnvs = seq.map { seq => memoizeValueIR(ctx, seq, seq.typ, memo) }
        val resEnv = memoizeValueIR(ctx, res, requestedType, memo)
        val valueType = unifySeq(
          TIterable.elementType(a.typ),
          uses(valueName, resEnv.eval) ++ seqEnvs.flatMap(e => uses(valueName, e.eval))
        )

        val accumNames = accum.map(_._1)
        val seqNames = accumNames ++ Array(valueName)
        unifyEnvsSeq(
          zeroEnvs
            ++ Array(resEnv.copy(eval = resEnv.eval.delete(accumNames)))
            ++ seqEnvs.map(e => e.copy(eval = e.eval.delete(seqNames)))
            ++ Array(memoizeValueIR(ctx, a, TStream(valueType), memo))
        )
      case StreamScan(a, zero, accumName, valueName, body) =>
        val zeroEnv = memoizeValueIR(ctx, zero, zero.typ, memo)
        val bodyEnv = memoizeValueIR(ctx, body, body.typ, memo)
        val valueType = unifySeq(TIterable.elementType(a.typ), uses(valueName, bodyEnv.eval))
        unifyEnvs(
          zeroEnv,
          bodyEnv.deleteEval(valueName).deleteEval(accumName),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
        
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val lElemType = TIterable.elementType(left.typ).asInstanceOf[TStruct]
        val rElemType = TIterable.elementType(right.typ).asInstanceOf[TStruct]

        val joinEnv = memoizeValueIR(ctx, join, TIterable.elementType(requestedType), memo)
        val lRequested = unifySeq(lElemType, uses(l, joinEnv.eval) :+ selectKey(lElemType, lKey))
        val rRequested = unifySeq(rElemType, uses(r, joinEnv.eval) :+ selectKey(rElemType, rKey))

        unifyEnvs(
          joinEnv.deleteEval(l).deleteEval(r),
          memoizeValueIR(ctx, left, TStream(lRequested), memo),
          memoizeValueIR(ctx, right, TStream(rRequested), memo)
        )
      case ArraySort(a, left, right, lessThan) =>
        val compEnv = memoizeValueIR(ctx, lessThan, lessThan.typ, memo)

        val requestedElementType = unifySeq(
          TIterable.elementType(a.typ),
          Array(TIterable.elementType(requestedType)) ++ uses(left, compEnv.eval) ++ uses(right, compEnv.eval)
        )
        
        unifyEnvs(
          compEnv.deleteEval(left).deleteEval(right),
          memoizeValueIR(ctx, a, TStream(requestedElementType), memo)
        )
        
      case ArrayMaximalIndependentSet(a, tiebreaker) =>
        tiebreaker.foreach { case (_, _, tb) => memoizeValueIR(ctx, tb, tb.typ, memo) }
        memoizeValueIR(ctx, a, a.typ, memo)
      case StreamFor(a, valueName, body) =>
        assert(requestedType == TVoid)
        val bodyEnv = memoizeValueIR(ctx, body, body.typ, memo)
        val valueType = unifySeq(
          TIterable.elementType(a.typ),
          uses(valueName, bodyEnv.eval))
        unifyEnvs(
          bodyEnv.deleteEval(valueName),
          memoizeValueIR(ctx, a, TStream(valueType), memo)
        )
      case MakeNDArray(data, shape, rowMajor, errorId) =>
        val elementType = requestedType.asInstanceOf[TNDArray].elementType
        val dataType = if (data.typ.isInstanceOf[TArray]) TArray(elementType) else TStream(elementType)
        unifyEnvs(
          memoizeValueIR(ctx, data, dataType, memo),
          memoizeValueIR(ctx, shape, shape.typ, memo),
          memoizeValueIR(ctx, rowMajor, rowMajor.typ, memo)
        )
      case NDArrayMap(nd, valueName, body) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        val bodyEnv = memoizeValueIR(ctx, body, requestedType.asInstanceOf[TNDArray].elementType, memo)
        val valueType = unifySeq(
          ndType.elementType,
          uses(valueName, bodyEnv.eval)
        )
        unifyEnvs(
          bodyEnv.deleteEval(valueName),
          memoizeValueIR(ctx, nd, ndType.copy(elementType = valueType), memo)
        )
      case NDArrayMap2(left, right, leftName, rightName, body, _) =>
        val leftType = left.typ.asInstanceOf[TNDArray]
        val rightType = right.typ.asInstanceOf[TNDArray]
        val bodyEnv = memoizeValueIR(ctx, body, requestedType.asInstanceOf[TNDArray].elementType, memo)

        val leftValueType = unifySeq(leftType.elementType, uses(leftName, bodyEnv.eval))
        val rightValueType = unifySeq(rightType.elementType, uses(rightName, bodyEnv.eval))

        unifyEnvs(
          bodyEnv.deleteEval(leftName).deleteEval(rightName),
          memoizeValueIR(ctx, left, leftType.copy(elementType = leftValueType), memo),
          memoizeValueIR(ctx, right, rightType.copy(elementType = rightValueType), memo)
        )
      case AggExplode(a, name, body, isScan) =>
        val bodyEnv = memoizeValueIR(ctx, body,
          requestedType,
          memo)
        if (isScan) {
          val valueType = unifySeq(TIterable.elementType(a.typ), uses(name, bodyEnv.scanOrEmpty))
          val aEnv = memoizeValueIR(ctx, a, TStream(valueType), memo)
          unifyEnvs(
            BindingEnv(scan = bodyEnv.scan.map(_.delete(name))),
            BindingEnv(scan = Some(aEnv.eval))
          )
        } else {
          val valueType = unifySeq(TIterable.elementType(a.typ), uses(name, bodyEnv.aggOrEmpty))
          val aEnv = memoizeValueIR(ctx, a, TStream(valueType), memo)
          unifyEnvs(
            BindingEnv(agg = bodyEnv.agg.map(_.delete(name))),
            BindingEnv(agg = Some(aEnv.eval))
          )
        }
      case AggFilter(cond, aggIR, isScan) =>
        val condEnv = memoizeValueIR(ctx, cond, cond.typ, memo)
        unifyEnvs(
          if (isScan)
            BindingEnv(scan = Some(condEnv.eval))
          else
            BindingEnv(agg = Some(condEnv.eval)),
          memoizeValueIR(ctx, aggIR, requestedType, memo)
        )
      case AggGroupBy(key, aggIR, isScan) =>
        val keyEnv = memoizeValueIR(ctx, key, requestedType.asInstanceOf[TDict].keyType, memo)
        unifyEnvs(
          if (isScan)
            BindingEnv(scan = Some(keyEnv.eval))
          else
            BindingEnv(agg = Some(keyEnv.eval)),
          memoizeValueIR(ctx, aggIR, requestedType.asInstanceOf[TDict].valueType, memo)
        )
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(ctx, aggBody,
          TIterable.elementType(requestedType),
          memo)
        if (isScan) {
          val valueType = unifySeq(TIterable.elementType(a.typ), uses(elementName, bodyEnv.scanOrEmpty))
          val aEnv = memoizeValueIR(ctx, a, TArray(valueType), memo)
          unifyEnvsSeq(FastSeq(
            bodyEnv.copy(eval = bodyEnv.eval.delete(indexName), scan = bodyEnv.scan.map(_.delete(elementName))),
            BindingEnv(scan = Some(aEnv.eval))
          ) ++ knownLength.map(x => memoizeValueIR(ctx, x, x.typ, memo)))
        } else {
          val valueType = unifySeq(TIterable.elementType(a.typ), uses(elementName, bodyEnv.aggOrEmpty))
          val aEnv = memoizeValueIR(ctx, a, TArray(valueType), memo)
          unifyEnvsSeq(FastSeq(
            bodyEnv.copy(eval = bodyEnv.eval.delete(indexName), agg = bodyEnv.agg.map(_.delete(elementName))),
            BindingEnv(agg = Some(aEnv.eval))
          ) ++ knownLength.map(x => memoizeValueIR(ctx, x, x.typ, memo)))
        }
      case ApplyAggOp(initOpArgs, seqOpArgs, sig) =>
        val prunedSig = AggSignature.prune(sig, requestedType)
        val initEnv = unifyEnvsSeq((initOpArgs, prunedSig.initOpArgs).zipped.map { (arg, req) => memoizeValueIR(ctx, arg, req, memo) })
        val seqOpEnv = unifyEnvsSeq((seqOpArgs, prunedSig.seqOpArgs).zipped.map { (arg, req) => memoizeValueIR(ctx, arg, req, memo) })
        BindingEnv(eval = initEnv.eval, agg = Some(seqOpEnv.eval))
      case ApplyScanOp(initOpArgs, seqOpArgs, sig) =>
        val prunedSig = AggSignature.prune(sig, requestedType)
        val initEnv = unifyEnvsSeq((initOpArgs, prunedSig.initOpArgs).zipped.map { (arg, req) => memoizeValueIR(ctx, arg, req, memo) })
        val seqOpEnv = unifyEnvsSeq((seqOpArgs, prunedSig.seqOpArgs).zipped.map { (arg, req) => memoizeValueIR(ctx, arg, req, memo) })
        BindingEnv(eval = initEnv.eval, scan = Some(seqOpEnv.eval))
      case AggFold(zero, seqOp, combOp, accumName, otherAccumName, isScan) =>
        val initEnv = memoizeValueIR(ctx, zero, zero.typ, memo)
        val seqEnv = memoizeValueIR(ctx, seqOp, seqOp.typ, memo)
        memoizeValueIR(ctx, combOp, combOp.typ, memo)

        if (isScan)
          BindingEnv(eval = initEnv.eval, scan = Some(seqEnv.eval.delete(accumName)))
        else
          BindingEnv(eval = initEnv.eval, agg = Some(seqEnv.eval.delete(accumName)))
      case StreamAgg(a, name, query) =>
        val queryEnv = memoizeValueIR(ctx, query, requestedType, memo)
        val requestedElemType = unifySeq(TIterable.elementType(a.typ), uses(name, queryEnv.aggOrEmpty))
        val aEnv = memoizeValueIR(ctx, a, TStream(requestedElemType), memo)
        unifyEnvs(
          BindingEnv(eval = concatEnvs(Array(queryEnv.eval, queryEnv.aggOrEmpty.delete(name)))),
          aEnv)
      case StreamAggScan(a, name, query) =>
        val queryEnv = memoizeValueIR(ctx, query, TIterable.elementType(requestedType), memo)
        val requestedElemType = unifySeq(
          TIterable.elementType(a.typ),
          uses(name, queryEnv.scanOrEmpty) ++ uses(name, queryEnv.eval)
        )
        val aEnv = memoizeValueIR(ctx, a, TStream(requestedElemType), memo)
        unifyEnvs(
          BindingEnv(eval = concatEnvs(Array(queryEnv.eval.delete(name), queryEnv.scanOrEmpty.delete(name)))),
          aEnv)
      case RunAgg(body, result, _) =>
        unifyEnvs(
          memoizeValueIR(ctx, body, body.typ, memo),
          memoizeValueIR(ctx, result, requestedType, memo)
        )
      case RunAggScan(array, name, init, seqs, result, signature) =>
        val resultEnv = memoizeValueIR(ctx, result, TIterable.elementType(requestedType), memo)
        val seqEnv = memoizeValueIR(ctx, seqs, seqs.typ, memo)
        val elemEnv = unifyEnvs(resultEnv, seqEnv)
        val requestedElemType = unifySeq(TIterable.elementType(array.typ), uses(name, elemEnv.eval))
        unifyEnvs(
          elemEnv,
          memoizeValueIR(ctx, array, TStream(requestedElemType), memo),
          memoizeValueIR(ctx, init, init.typ, memo)
        )
      case MakeStruct(fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        unifyEnvsSeq(fields.flatMap { case (fname, fir) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          sType.fieldOption(fname).map(f => memoizeValueIR(ctx, fir, f.typ, memo))
        })
      case InsertFields(old, fields, _) =>
        val sType = requestedType.asInstanceOf[TStruct]
        val insFieldNames = fields.map(_._1).toSet
        val rightDep = sType.filter(f => insFieldNames.contains(f.name))._1
        val rightDepFields = rightDep.fieldNames.toSet
        val leftDep = TStruct(
          old.typ.asInstanceOf[TStruct]
            .fields
            .flatMap { f =>
              if (rightDep.hasField(f.name))
                Some(f.name -> minimal(f.typ))
              else
                sType.fieldOption(f.name).map(f.name -> _.typ)
            }: _*)
        unifyEnvsSeq(
          FastSeq(memoizeValueIR(ctx, old, leftDep, memo)) ++
            // ignore unreachable fields, these are eliminated on the upwards pass
            fields.flatMap { case (fname, fir) =>
              rightDep.fieldOption(fname).map(f => memoizeValueIR(ctx, fir, f.typ, memo))
            }
        )
      case SelectFields(old, fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        val oldReqType = TStruct(old.typ.asInstanceOf[TStruct]
          .fieldNames
          .flatMap(fn => sType.fieldOption(fn).map(fd => (fd.name, fd.typ))): _*)
        memoizeValueIR(ctx, old, oldReqType, memo)
      case GetField(o, name) =>
        memoizeValueIR(ctx, o, TStruct(name -> requestedType), memo)
      case MakeTuple(fields) =>
        val tType = requestedType.asInstanceOf[TTuple]

        unifyEnvsSeq(
          fields.flatMap { case (i, value) =>
            // ignore unreachable fields, these are eliminated on the upwards pass
            tType.fieldIndex.get(i)
              .map { idx =>
                memoizeValueIR(ctx, value, tType.types(idx), memo)
              }})
      case GetTupleElement(o, idx) =>
        val childTupleType = o.typ.asInstanceOf[TTuple]
        val tupleDep = TTuple(FastSeq(TupleField(idx, requestedType)))
        memoizeValueIR(ctx, o, tupleDep, memo)
      case ConsoleLog(message, result) =>
        unifyEnvs(
          memoizeValueIR(ctx, message, TString, memo),
          memoizeValueIR(ctx, result, result.typ, memo)
        )
      case MatrixCount(child) =>
        memoizeMatrixIR(ctx, child, minimal(child.typ), memo)
        BindingEnv.empty
      case TableCount(child) =>
        memoizeTableIR(ctx, child, minimal(child.typ), memo)
        BindingEnv.empty
      case TableGetGlobals(child) =>
        memoizeTableIR(ctx, child, minimal(child.typ).copy(globalType = requestedType.asInstanceOf[TStruct]), memo)
        BindingEnv.empty
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        memoizeTableIR(ctx, child, TableType(
          key = child.typ.key,
          rowType = unify(child.typ.rowType,
            rStruct.fieldOption("rows").map(_.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]).getOrElse(TStruct.empty)),
          globalType = rStruct.fieldOption("global").map(_.typ.asInstanceOf[TStruct]).getOrElse(TStruct.empty)),
          memo)
        BindingEnv.empty
      case TableToValueApply(child, _) =>
        memoizeTableIR(ctx, child, child.typ, memo)
        BindingEnv.empty
      case MatrixToValueApply(child, _) => memoizeMatrixIR(ctx, child, child.typ, memo)
        BindingEnv.empty
      case BlockMatrixToValueApply(child, _) => memoizeBlockMatrixIR(ctx, child, child.typ, memo)
        BindingEnv.empty
      case TableAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(ctx, query, query.typ, child.typ, memo)
        val dep = TableType(
          key = child.typ.key,
          rowType = unify(child.typ.rowType, queryDep.rowType, selectKey(child.typ.rowType, child.typ.key)),
          globalType = queryDep.globalType
        )
        memoizeTableIR(ctx, child, dep, memo)
        BindingEnv.empty
      case MatrixAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(ctx, query, query.typ, child.typ, memo)
        val dep = MatrixType(
          rowKey = child.typ.rowKey,
          colKey = FastSeq(),
          rowType = unify(child.typ.rowType, queryDep.rowType, selectKey(child.typ.rowType, child.typ.rowKey)),
          entryType = queryDep.entryType,
          colType = queryDep.colType,
          globalType = queryDep.globalType
        )
        memoizeMatrixIR(ctx, child, dep, memo)
        BindingEnv.empty
      case TailLoop(name, params, body) =>
        val bodyEnv = memoizeValueIR(ctx, body, body.typ, memo)
        val paramTypes = params.map{ case (paramName, paramIR) =>
          unifySeq(paramIR.typ, uses(paramName, bodyEnv.eval))
        }
        unifyEnvsSeq(
          IndexedSeq(bodyEnv.deleteEval(params.map(_._1))) ++
            (params, paramTypes).zipped.map{ case ((paramName, paramIR), paramType) =>
            memoizeValueIR(ctx, paramIR, paramType, memo)
          }
        )
      case CollectDistributedArray(contexts, globals, cname, gname, body, dynamicID, _, tsd) =>
        val rArray = requestedType.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(ctx, body, rArray.elementType, memo)
        assert(bodyEnv.scan.isEmpty)
        assert(bodyEnv.agg.isEmpty)

        val cDep = unifySeq(TIterable.elementType(contexts.typ), uses(cname, bodyEnv.eval))
        val gDep = unifySeq(globals.typ, uses(gname, bodyEnv.eval))

        unifyEnvs(
          memoizeValueIR(ctx, contexts, TStream(cDep), memo),
          memoizeValueIR(ctx, globals, gDep, memo),
          memoizeValueIR(ctx, dynamicID, TString, memo),
        )
      case _: IR =>
        val envs = ir.children.flatMap {
          case mir: MatrixIR =>
            memoizeMatrixIR(ctx, mir, mir.typ, memo)
            None
          case tir: TableIR =>
            memoizeTableIR(ctx, tir, tir.typ, memo)
            None
          case bmir: BlockMatrixIR => //NOTE Currently no BlockMatrixIRs would have dead fields
            None
          case ir: IR =>
            Some(memoizeValueIR(ctx, ir, ir.typ, memo))
        }
        unifyEnvsSeq(envs.toFastSeq)
    }
  }

  def rebuild(
    ctx: ExecuteContext,
    tir: TableIR,
    memo: RebuildMutableState
  ): TableIR = {
    val requestedType = memo.requestedType.lookup(tir).asInstanceOf[TableType]
    tir match {
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        TableParallelize(
          upcast(ctx, rebuildIR(ctx, rowsAndGlobal, BindingEnv.empty, memo),
            memo.requestedType.lookup(rowsAndGlobal).asInstanceOf[TStruct]),
          nPartitions)

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
          errorId
        )

      case TableRead(typ, dropRows, tr) =>
        // FIXME: remove this when all readers know how to read without keys
        val requestedTypeWithKey = TableType(
          key = typ.key,
          rowType = unify(typ.rowType, selectKey(typ.rowType, typ.key), requestedType.rowType),
          globalType = requestedType.globalType)
        TableRead(requestedTypeWithKey, dropRows, tr)
      case TableFilter(child, pred) =>
        val child2 = rebuild(ctx, child, memo)
        val pred2 = rebuildIR(ctx, pred, BindingEnv(child2.typ.rowEnv), memo)
        TableFilter(child2, pred2)
      case TableMapPartitions(child, gName, pName, body, requestedKey, allowedOverlap) =>
        val child2 = rebuild(ctx, child, memo)
        val body2 = rebuildIR(ctx, body, BindingEnv(Env(
          gName -> child2.typ.globalType,
          pName -> TStream(child2.typ.rowType))), memo)
        val body2ElementType = TIterable.elementType(body2.typ).asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.key.exists(k => !body2ElementType.hasField(k)))
          TableKeyBy(child2, child2.typ.key.takeWhile(body2ElementType.hasField))
        else
          child2
        val childKeyLen = child2Keyed.typ.key.length
        require(requestedKey <= childKeyLen)
        TableMapPartitions(child2Keyed, gName, pName, body2, requestedKey, math.min(allowedOverlap, childKeyLen))
      case TableMapRows(child, newRow) =>
        val child2 = rebuild(ctx, child, memo)
        val newRow2 = rebuildIR(ctx, newRow, BindingEnv(child2.typ.rowEnv, scan = Some(child2.typ.rowEnv)), memo)
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
          child2 = upcastTable(ctx, child2, memo.requestedType.lookup(child).asInstanceOf[TableType], upcastGlobals = false)
        TableKeyBy(child2, keys2, isSorted)
      case TableOrderBy(child, sortFields) =>
        val child2 = if (sortFields.forall(_.sortOrder == Ascending) && child.typ.key.startsWith(sortFields.map(_.field)))
          rebuild(ctx, child, memo)
        else {
          // fully upcast before shuffle
          upcastTable(ctx, rebuild(ctx, child, memo), memo.requestedType.lookup(child).asInstanceOf[TableType], upcastGlobals = false)
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
        val rebuilt = children.map { c => rebuild(ctx, c, memo) }
        val upcasted = rebuilt.map { t => upcastTable(ctx, t, memo.requestedType.lookup(children(0)).asInstanceOf[TableType]) }
        TableMultiWayZipJoin(upcasted, fieldName, globalName)
      case TableAggregateByKey(child, expr) =>
        val child2 = rebuild(ctx, child, memo)
        TableAggregateByKey(child2, rebuildIR(ctx, expr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo))
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val child2 = rebuild(ctx, child, memo)
        val expr2 = rebuildIR(ctx, expr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo)
        val newKey2 = rebuildIR(ctx, newKey, BindingEnv(child2.typ.rowEnv), memo)
        TableKeyByAndAggregate(child2, expr2, newKey2, nPartitions, bufferSize)
      case TableRename(child, rowMap, globalMap) =>
        val child2 = rebuild(ctx, child, memo)
        TableRename(
          child2,
          rowMap.filterKeys(child2.typ.rowType.hasField),
          globalMap.filterKeys(child2.typ.globalType.hasField))
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
    memo: RebuildMutableState
  ): MatrixIR = {
    val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
    mir match {
      case x@MatrixRead(typ, dropCols, dropRows, reader) =>
        // FIXME: remove this when all readers know how to read without keys
        val requestedTypeWithKeys = MatrixType(
          rowKey = typ.rowKey,
          colKey = typ.colKey,
          rowType = unify(typ.rowType, selectKey(typ.rowType, typ.rowKey), requestedType.rowType),
          entryType = requestedType.entryType,
          colType = unify(typ.colType, selectKey(typ.colType, typ.colKey), requestedType.colType),
          globalType = requestedType.globalType
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
        val newRow2 = rebuildIR(ctx, newRow,
          BindingEnv(child2.typ.rowEnv, agg = Some(child2.typ.entryEnv), scan = Some(child2.typ.rowEnv)), memo)
        val newRowType = newRow2.typ.asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.rowKey.exists(k => !newRowType.hasField(k)))
          MatrixKeyRowsBy(child2, child2.typ.rowKey.takeWhile(newRowType.hasField))
        else
          child2
        MatrixMapRows(child2Keyed, newRow2)
      case MatrixMapCols(child, newCol, newKey) =>
        val child2 = rebuild(ctx, child, memo)
        val newCol2 = rebuildIR(ctx, newCol,
          BindingEnv(child2.typ.colEnv, agg = Some(child2.typ.entryEnv), scan = Some(child2.typ.colEnv)), memo)
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
        MatrixAggregateRowsByKey(child2,
          rebuildIR(ctx, entryExpr, BindingEnv(child2.typ.colEnv, agg = Some(child2.typ.entryEnv)), memo),
          rebuildIR(ctx, rowExpr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo))
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val child2 = rebuild(ctx, child, memo)
        MatrixAggregateColsByKey(child2,
          rebuildIR(ctx, entryExpr, BindingEnv(child2.typ.rowEnv, agg = Some(child2.typ.entryEnv)), memo),
          rebuildIR(ctx, colExpr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.colEnv)), memo))
      case MatrixUnionRows(children) =>
        val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
        val firstChild = upcast(ctx, rebuild(ctx, children.head, memo), requestedType, upcastGlobals = false)
        val remainingChildren = children.tail.map { child =>
          upcast(ctx, rebuild(ctx, child, memo), requestedType.copy(colType = requestedType.colKeyStruct),
            upcastGlobals = false)
        }
        MatrixUnionRows(firstChild +: remainingChildren)
      case MatrixUnionCols(left, right, joinType) =>
        val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
        val left2 = rebuild(ctx, left, memo)
        val right2 = rebuild(ctx, right, memo)

        if (left2.typ.colType == right2.typ.colType && left2.typ.entryType == right2.typ.entryType) {
          MatrixUnionCols(
            left2,
            right2,
            joinType
          )
        } else {
          MatrixUnionCols(
            upcast(ctx, left2, requestedType, upcastRows=false, upcastGlobals = false),
            upcast(ctx, right2, requestedType, upcastRows=false, upcastGlobals = false),
            joinType
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
          globalMap.filterKeys(child2.typ.globalType.hasField),
          colMap.filterKeys(child2.typ.colType.hasField),
          rowMap.filterKeys(child2.typ.rowType.hasField),
          entryMap.filterKeys(child2.typ.entryType.hasField))
      case RelationalLetMatrixTable(name, value, body) =>
        val value2 = rebuildIR(ctx, value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLetMatrixTable(name, value2, rebuild(ctx, body, memo))
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        CastTableToMatrix(rebuild(ctx, child, memo), entriesFieldName, colsFieldName, requestedType.colKey)
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
    memo: RebuildMutableState
  ): BlockMatrixIR = bmir match {
    case RelationalLetBlockMatrix(name, value, body) =>
      val value2 = rebuildIR(ctx, value, BindingEnv.empty, memo)
      memo.relationalRefs += name -> value2.typ
      RelationalLetBlockMatrix(name, value2, rebuild(ctx, body, memo))
    case _ =>
      bmir.mapChildren {
        case tir: TableIR => rebuild(ctx, tir, memo)
        case mir: MatrixIR => rebuild(ctx, mir, memo)
        case ir: IR => rebuildIR(ctx, ir, BindingEnv.empty[Type], memo)
        case bmir: BlockMatrixIR => rebuild(ctx, bmir, memo)
      }.asInstanceOf[BlockMatrixIR]
  }

  def rebuildIR(
    ctx: ExecuteContext,
    ir: IR,
    env: BindingEnv[Type],
    memo: RebuildMutableState
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
              val castFields = cast.map { f => f.index -> f.typ }.toMap
              val baseFields = base.map { f => f.index -> f.typ }.toMap
              TTuple(reb.map { f => TupleField(f.index, recur(f.typ, castFields(f.index), baseFields(f.index)))})
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
          If(cond2,
            upcast(ctx, cnsq2, requestedType),
            upcast(ctx, alt2, requestedType)
          )
      case Coalesce(values) =>
        val values2 = values.map(rebuildIR(ctx, _, env, memo))
        require(values2.nonEmpty)
        if (values2.forall(_.typ == values2.head.typ))
          Coalesce(values2)
        else
          Coalesce(values2.map(upcast(ctx, _, requestedType)))
      case Consume(value) =>
        val value2 = rebuildIR(ctx, value, env, memo)
        Consume(value2)
      case Let(name, value, body) =>
        val value2 = rebuildIR(ctx, value, env, memo)
        Let(
          name,
          value2,
          rebuildIR(ctx, body, env.bindEval(name, value2.typ), memo)
        )
      case AggLet(name, value, body, isScan) =>
        val value2 = rebuildIR(ctx, value, if (isScan) env.promoteScan else env.promoteAgg, memo)
        AggLet(
          name,
          value2,
          rebuildIR(ctx, body, if (isScan) env.bindScan(name, value2.typ) else env.bindAgg(name, value2.typ), memo),
          isScan
        )
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
        MakeStream.unify(ctx, args2, requiresMemoryManagementPerElement, requestedType = TStream(dep.elementType))
      case StreamMap(a, name, body) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        StreamMap(a2, name, rebuildIR(ctx, body, env.bindEval(name, TIterable.elementType(a2.typ)), memo))
      case StreamZip(as, names, body, b, errorID) =>
        val (newAs, newNames) = (as, names)
          .zipped
          .flatMap { case (a, name) => if (memo.requestedType.contains(a)) Some((rebuildIR(ctx, a, env, memo), name)) else None }
          .unzip
        StreamZip(newAs, newNames, rebuildIR(ctx, body,
          env.bindEval(newNames.zip(newAs.map(a => TIterable.elementType(a.typ))): _*), memo), b, errorID)
      case StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val newAs = as.map(a => rebuildIR(ctx, a, env, memo))
        val newEltType = TIterable.elementType(as.head.typ).asInstanceOf[TStruct]
        val newJoinF = rebuildIR(ctx,
          joinF,
          env.bindEval(curKey -> selectKey(newEltType, key), curVals -> TArray(newEltType)),
          memo)
        StreamZipJoin(newAs, key, curKey, curVals, newJoinF)
      case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, joinF) =>
        val newContexts = rebuildIR(ctx, contexts, env, memo)
        val newCtxType = TIterable.elementType(newContexts.typ)
        val newMakeProducer = rebuildIR(ctx, makeProducer, env.bindEval(ctxName, newCtxType), memo)
        val newEltType = TIterable.elementType(newMakeProducer.typ).asInstanceOf[TStruct]
        val newJoinF = rebuildIR(ctx,
          joinF,
          env.bindEval(curKey -> selectKey(newEltType, key), curVals -> TArray(newEltType)),
          memo)
        StreamZipJoinProducers(newContexts, ctxName, newMakeProducer,key, curKey, curVals, newJoinF)
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
      case StreamFilter(a, name, cond) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        StreamFilter(a2, name, rebuildIR(ctx, cond, env.bindEval(name, TIterable.elementType(a2.typ)), memo))
      case StreamTakeWhile(a, name, cond) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        StreamTakeWhile(a2, name, rebuildIR(ctx, cond, env.bindEval(name, TIterable.elementType(a2.typ)), memo))
      case StreamDropWhile(a, name, cond) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        StreamDropWhile(a2, name, rebuildIR(ctx, cond, env.bindEval(name, TIterable.elementType(a2.typ)), memo))
      case StreamFlatMap(a, name, body) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        StreamFlatMap(a2, name, rebuildIR(ctx, body, env.bindEval(name, TIterable.elementType(a2.typ)), memo))
      case StreamFold(a, zero, accumName, valueName, body) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val z2 = rebuildIR(ctx, zero, env, memo)
        StreamFold(
          a2,
          z2,
          accumName,
          valueName,
          rebuildIR(ctx, body, env.bindEval(accumName -> z2.typ, valueName -> TIterable.elementType(a2.typ)), memo)
        )
      case StreamFold2(a: IR, accum, valueName, seqs, result) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val newAccum = accum.map { case (n, z) => n -> rebuildIR(ctx, z, env, memo) }
        val newEnv = env
          .bindEval(valueName -> TIterable.elementType(a2.typ))
          .bindEval(newAccum.map { case (n, z) => n -> z.typ }: _*)
        StreamFold2(
          a2,
          newAccum,
          valueName,
          seqs.map(rebuildIR(ctx, _, newEnv, memo)),
          rebuildIR(ctx, result, newEnv, memo))
      case StreamScan(a, zero, accumName, valueName, body) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val z2 = rebuildIR(ctx, zero, env, memo)
        StreamScan(
          a2,
          z2,
          accumName,
          valueName,
          rebuildIR(ctx, body, env.bindEval(accumName -> z2.typ, valueName -> TIterable.elementType(a2.typ)), memo)
        )
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val left2 = rebuildIR(ctx, left, env, memo)
        val right2 = rebuildIR(ctx, right, env, memo)

        val ltyp = left2.typ.asInstanceOf[TStream]
        val rtyp = right2.typ.asInstanceOf[TStream]
        StreamJoinRightDistinct(
          left2, right2, lKey, rKey, l, r,
          rebuildIR(ctx, join, env.bindEval(l -> ltyp.elementType, r -> rtyp.elementType), memo),
          joinType)
      case StreamFor(a, valueName, body) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val body2 = rebuildIR(ctx, body, env.bindEval(valueName -> TIterable.elementType(a2.typ)), memo)
        StreamFor(a2, valueName, body2)
      case ArraySort(a, left, right, lessThan) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val et = TIterable.elementType(a2.typ)
        val lessThan2 = rebuildIR(ctx, lessThan, env.bindEval(left -> et, right -> et), memo)
        ArraySort(a2, left, right, lessThan2)
      case MakeNDArray(data, shape, rowMajor, errorId) =>
        val data2 = rebuildIR(ctx, data, env, memo)
        val shape2 = rebuildIR(ctx, shape, env, memo)
        val rowMajor2 = rebuildIR(ctx, rowMajor, env, memo)
        MakeNDArray(data2, shape2, rowMajor2, errorId)
      case NDArrayMap(nd, valueName, body) =>
        val nd2 = rebuildIR(ctx, nd, env, memo)
        NDArrayMap(nd2, valueName, rebuildIR(ctx, body, env.bindEval(valueName, nd2.typ.asInstanceOf[TNDArray].elementType), memo))
      case NDArrayMap2(left, right, leftName, rightName, body, errorID) =>
        val left2 = rebuildIR(ctx, left, env, memo)
        val right2 = rebuildIR(ctx, right, env, memo)
        val body2 = rebuildIR(ctx, body,
          env.bindEval(leftName, left2.typ.asInstanceOf[TNDArray].elementType).bindEval(rightName, right2.typ.asInstanceOf[TNDArray].elementType),
          memo)
        NDArrayMap2(left2, right2, leftName, rightName, body2, errorID)
      case MakeStruct(fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        // drop unnecessary field IRs
        val depFields = depStruct.fieldNames.toSet
        MakeStruct(fields.flatMap { case (f, fir) =>
          if (depFields.contains(f))
            Some(f -> rebuildIR(ctx, fir, env, memo))
          else {
            log.info(s"Prune: MakeStruct: eliminating field '$f'")
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

        val insertOverwritesUnrequestedButPreservedField = fields.exists{ case (fieldName, _) =>
          preservedChildFields.contains(fieldName) && !depFields.contains(fieldName)
        }

        val wrappedChild = if (insertOverwritesUnrequestedButPreservedField) {
          val selectedChildFields = preservedChildFields.filter(s => depFields.contains(s))
          SelectFields(rebuiltChild, rebuiltChild.typ.asInstanceOf[TStruct].fieldNames.filter(selectedChildFields.contains(_)))
        } else {
          rebuiltChild
        }

        InsertFields(wrappedChild,
          fields.flatMap { case (f, fir) =>
            if (depFields.contains(f))
              Some(f -> rebuildIR(ctx, fir, env, memo))
            else {
              log.info(s"Prune: InsertFields: eliminating field '$f'")
              None
            }
          }, fieldOrder.map(fds => fds.filter(f => depFields.contains(f) || wrappedChild.typ.asInstanceOf[TStruct].hasField(f))))
      case SelectFields(old, fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val old2 = rebuildIR(ctx, old, env, memo)
        SelectFields(old2, fields.filter(f => old2.typ.asInstanceOf[TStruct].hasField(f) && depStruct.hasField(f)))
      case ConsoleLog(message, result) =>
        val message2 = rebuildIR(ctx, message, env, memo)
        val result2 = rebuildIR(ctx, result, env, memo)
        ConsoleLog(message2, result2)
      case TableAggregate(child, query) =>
        val child2 = rebuild(ctx, child, memo)
        val query2 = rebuildIR(ctx, query, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo)
        TableAggregate(child2, query2)
      case MatrixAggregate(child, query) =>
        val child2 = rebuild(ctx, child, memo)
        val query2 = rebuildIR(ctx, query, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.entryEnv)), memo)
        MatrixAggregate(child2, query2)
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        if (!rStruct.hasField("rows"))
          if (rStruct.hasField("global"))
            MakeStruct(FastSeq("global" -> TableGetGlobals(rebuild(ctx, child, memo))))
          else
            MakeStruct(FastSeq())
        else {
          val rRowType = TIterable.elementType(rStruct.fieldType("rows")).asInstanceOf[TStruct]
          val rGlobType = rStruct.fieldOption("global").map(_.typ.asInstanceOf[TStruct]).getOrElse(TStruct())
          TableCollect(upcastTable(ctx, rebuild(ctx, child, memo), TableType(rowType = rRowType, FastSeq(), rGlobType),
            upcastRow = true, upcastGlobals = false))
        }
      case AggExplode(array, name, aggBody, isScan) =>
        val a2 = rebuildIR(ctx, array, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val a2t = TIterable.elementType(a2.typ)
        val body2 = rebuildIR(ctx, aggBody, if (isScan) env.bindScan(name, a2t) else env.bindAgg(name, a2t), memo)
        AggExplode(a2, name, body2, isScan)
      case AggFilter(cond, aggIR, isScan) =>
        val cond2 = rebuildIR(ctx, cond, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val aggIR2 = rebuildIR(ctx, aggIR, env, memo)
        AggFilter(cond2, aggIR2, isScan)
      case AggGroupBy(key, aggIR, isScan) =>
        val key2 = rebuildIR(ctx, key, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val aggIR2 = rebuildIR(ctx, aggIR, env, memo)
        AggGroupBy(key2, aggIR2, isScan)
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) =>
        val aEnv = if (isScan) env.promoteScan else env.promoteAgg
        val a2 = rebuildIR(ctx, a, aEnv, memo)
        val a2t = TIterable.elementType(a2.typ)
        val env_ = env.bindEval(indexName -> TInt32)
        val aggBody2 = rebuildIR(ctx, aggBody, if (isScan) env_.bindScan(elementName, a2t) else env_.bindAgg(elementName, a2t), memo)
        AggArrayPerElement(a2, elementName, indexName, aggBody2, knownLength.map(rebuildIR(ctx, _, aEnv, memo)), isScan)
      case StreamAgg(a, name, query) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val newEnv = env.copy(agg = Some(env.eval.bind(name -> TIterable.elementType(a2.typ))))
        val query2 = rebuildIR(ctx, query, newEnv, memo)
        StreamAgg(a2, name, query2)
      case StreamAggScan(a, name, query) =>
        val a2 = rebuildIR(ctx, a, env, memo)
        val query2 = rebuildIR(ctx, query, env.copy(scan = Some(env.eval.bind(name -> TIterable.elementType(a2.typ)))), memo)
        StreamAggScan(a2, name, query2)
      case RunAgg(body, result, signatures) =>
        val body2 = rebuildIR(ctx, body, env, memo)
        val result2 = rebuildIR(ctx, result, env, memo)
        RunAgg(body2, result2, signatures)
      case RunAggScan(array, name, init, seqs, result, signature) =>
        val array2 = rebuildIR(ctx, array, env, memo)
        val init2 = rebuildIR(ctx, init, env, memo)
        val eltEnv = env.bindEval(name, TIterable.elementType(array2.typ))
        val seqs2 = rebuildIR(ctx, seqs, eltEnv, memo)
        val result2 = rebuildIR(ctx, result, eltEnv, memo)
        RunAggScan(array2, name, init2, seqs2, result2, signature)
      case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
        val initOpArgs2 = initOpArgs.map(rebuildIR(ctx, _, env, memo))
        val seqOpArgs2 = seqOpArgs.map(rebuildIR(ctx, _, env.promoteAgg, memo))
        ApplyAggOp(initOpArgs2, seqOpArgs2,
          aggSig.copy(
            initOpArgs = initOpArgs2.map(_.typ),
            seqOpArgs = seqOpArgs2.map(_.typ)))
      case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
        val initOpArgs2 = initOpArgs.map(rebuildIR(ctx, _, env, memo))
        val seqOpArgs2 = seqOpArgs.map(rebuildIR(ctx, _, env.promoteScan, memo))
        ApplyScanOp(initOpArgs2, seqOpArgs2,
          aggSig.copy(
            initOpArgs = initOpArgs2.map(_.typ),
            seqOpArgs = seqOpArgs2.map(_.typ)))
      case AggFold(zero, seqOp, combOp, accumName, otherAccumName, isScan) =>
        val zero2 = rebuildIR(ctx, zero, env, memo)
        val seqOp2 = rebuildIR(ctx, seqOp, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val combOp2 = rebuildIR(ctx, combOp, env, memo)
        AggFold(zero2, seqOp2, combOp2, accumName, otherAccumName, isScan)
      case CollectDistributedArray(contexts, globals, cname, gname, body, dynamicID, staticID, tsd) =>
        val contexts2 = upcast(ctx, rebuildIR(ctx, contexts, env, memo), memo.requestedType.lookup(contexts).asInstanceOf[Type])
        val globals2 = upcast(ctx, rebuildIR(ctx, globals, env, memo), memo.requestedType.lookup(globals).asInstanceOf[Type])
        val body2 = rebuildIR(ctx, body, BindingEnv(Env(cname -> TIterable.elementType(contexts2.typ), gname -> globals2.typ)), memo)
        val dynamicID2 = rebuildIR(ctx, dynamicID, env, memo)
        CollectDistributedArray(contexts2, globals2, cname, gname, body2, dynamicID2, staticID, tsd)
      case _ =>
        ir.mapChildren {
          case valueIR: IR => rebuildIR(ctx, valueIR, env, memo) // FIXME: assert IR does not bind or change env
          case mir: MatrixIR => rebuild(ctx, mir, memo)
          case tir: TableIR => rebuild(ctx, tir, memo)
          case bmir: BlockMatrixIR => bmir //NOTE Currently no BlockMatrixIRs would have dead fields
        }
    }
  }

  def upcast(ctx: ExecuteContext, ir: IR, rType: Type): IR = {
    if (ir.typ == rType)
      ir
    else {
      val result = ir.typ match {
        case _: TStruct =>
          val rs = rType.asInstanceOf[TStruct]
          val uid = genUID()
          val ref = Ref(uid, ir.typ)
          val ms = MakeStruct(
            rs.fields.map { f =>
              f.name -> upcast(ctx, GetField(ref, f.name), f.typ)
            }
          )
          Let(uid, ir, If(IsNA(ref), NA(ms.typ), ms))
        case ts: TStream =>
          val ra = rType.asInstanceOf[TStream]
          val uid = genUID()
          val ref = Ref(uid, ts.elementType)
          StreamMap(ir, uid, upcast(ctx, ref, ra.elementType))
        case ts: TArray =>
          val ra = rType.asInstanceOf[TArray]
          val uid = genUID()
          val ref = Ref(uid, ts.elementType)
          ToArray(StreamMap(ToStream(ir), uid, upcast(ctx, ref, ra.elementType)))
        case _: TTuple =>
          val rt = rType.asInstanceOf[TTuple]
          val uid = genUID()
          val ref = Ref(uid, ir.typ)
          val mt = MakeTuple(rt._types.map { tupleField =>
            tupleField.index -> upcast(ctx, GetTupleElement(ref, tupleField.index), tupleField.typ)
          })
          Let(uid, ir, If(IsNA(ref), NA(mt.typ), mt))
        case _: TDict =>
          val rd = rType.asInstanceOf[TDict]
          ToDict(upcast(ctx, ToStream(ir), TArray(rd.elementType)))
        case _: TSet =>
          val rs = rType.asInstanceOf[TSet]
          ToSet(upcast(ctx, ToStream(ir), TSet(rs.elementType)))
        case _ => ir
      }

      assert(result.typ == rType, s"${ Pretty(ctx, result) }, ${ result.typ }, $rType")
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
    upcastEntries: Boolean = true
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
        mt = MatrixMapEntries(mt, upcast(ctx, Ref("g", mt.typ.entryType), rType.entryType))

      if (upcastRows && mt.typ.rowType != rType.rowType)
        mt = MatrixMapRows(mt, upcast(ctx, Ref("va", mt.typ.rowType), rType.rowType))

      if (upcastCols && (mt.typ.colType != rType.colType || mt.typ.colKey != rType.colKey)) {
        mt = MatrixMapCols(mt, upcast(ctx, Ref("sa", mt.typ.colType), rType.colType),
          if (rType.colKey == mt.typ.colKey) None else Some(rType.colKey))
      }

      if (upcastGlobals && mt.typ.globalType != rType.globalType)
        mt = MatrixMapGlobals(mt, upcast(ctx, Ref("global", ir.typ.globalType), rType.globalType))

      mt
    }
  }

  def upcastTable(
    ctx: ExecuteContext,
    ir: TableIR,
    rType: TableType,
    upcastRow: Boolean = true,
    upcastGlobals: Boolean = true
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
        table = TableMapRows(table, upcast(ctx, Ref("row", table.typ.rowType), rType.rowType))
      }
      if (upcastGlobals && ir.typ.globalType != rType.globalType) {
        table = TableMapGlobals(table,
          upcast(ctx, Ref("global", table.typ.globalType), rType.globalType))
      }
      table
    }
  }
}
