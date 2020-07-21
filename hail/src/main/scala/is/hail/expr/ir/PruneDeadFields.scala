package is.hail.expr.ir

import is.hail.annotations._
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable


object PruneDeadFields {

  case class ComputeMutableState(requestedType: Memo[BaseType], relationalRefs: mutable.HashMap[String, ArrayBuilder[Type]]) {
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
            t1.size == t2.size &&
            t1.types.zip(t2.types)
              .forall { case (elt1, elt2) => isSupertype(elt1, elt2) }
        case (t1: Type, t2: Type) => t1 == t2
        case _ => fatal(s"invalid comparison: $superType / $subType")
      }
    } catch {
      case e: Throwable =>
        fatal(s"error while checking subtype:\n  super: $superType\n  sub:   $subType", e)
    }
  }

  def apply(ir: BaseIR): BaseIR = {
    try {
      val irCopy = ir.deepCopy()
      val ms = ComputeMutableState(Memo.empty[BaseType], mutable.HashMap.empty)
      irCopy match {
        case mir: MatrixIR =>
          memoizeMatrixIR(mir, mir.typ, ms)
          rebuild(mir, ms.rebuildState)
        case tir: TableIR =>
          memoizeTableIR(tir, tir.typ, ms)
          rebuild(tir, ms.rebuildState)
        case bmir: BlockMatrixIR =>
          memoizeBlockMatrixIR(bmir, bmir.typ, ms)
          rebuild(bmir, ms.rebuildState)
        case vir: IR =>
          memoizeValueIR(vir, vir.typ, ms)
          rebuildIR(vir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)), ms.rebuildState)
      }
    } catch {
      case e: Throwable => fatal(s"error trying to rebuild IR:\n${ Pretty(ir, elideLiterals = true) }", e)
    }
  }

  def selectKey(t: TStruct, k: IndexedSeq[String]): TStruct = t.filterSet(k.toSet)._1

  def minimal(tt: TableType): TableType = {
    TableType(
      rowType = TStruct.empty,
      key = FastIndexedSeq(),
      globalType = TStruct.empty
    )
  }

  def minimal(mt: MatrixType): MatrixType = {
    MatrixType(
      rowKey = FastIndexedSeq(),
      colKey = FastIndexedSeq(),
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
    children.foldLeft(FastIndexedSeq[String]()) { case (comb, k) => if (k.length > comb.length) k else comb }
  }

  def unifyBaseType(base: BaseType, children: BaseType*): BaseType = unifyBaseTypeSeq(base, children)

  def unifyBaseTypeSeq(base: BaseType, children: Seq[BaseType]): BaseType = {
    try {
      if (children.isEmpty)
        return minimalBT(base)
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
              val subFields = ts.fields.map { f =>
                f -> subStructs.flatMap(s => s.fieldOption(f.name))
              }
                .filter(_._2.nonEmpty)
                .map { case (f, ss) => f.name -> unifySeq(f.typ, ss.map(_.typ)) }
              TStruct(subFields: _*)
            case tt: TTuple =>
              val subTuples = children.map(_.asInstanceOf[TTuple])
              TTuple(tt._types.map { fd => fd -> subTuples.flatMap(child => child.fieldIndex.get(fd.index).map(child.types)) }
                .filter(_._2.nonEmpty)
                .map { case (fd, fdChildren) => TupleField(fd.index, unifySeq(fd.typ, fdChildren)) })
            case ta: TArray =>
              TArray(unifySeq(ta.elementType, children.map(_.asInstanceOf[TArray].elementType)))
            case ts: TStream =>
              TStream(unifySeq(ts.elementType, children.map(_.asInstanceOf[TStream].elementType)))
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
        throw new RuntimeException(s"failed to unify children while unifying:\n  base:  ${ base }\n${ children.mkString("\n") }", e)
    }
  }

  def unify[T <: BaseType](base: T, children: T*): T = unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def unifySeq[T <: BaseType](base: T, children: Seq[T]): T = unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def unifyEnvs(envs: BindingEnv[ArrayBuilder[Type]]*): BindingEnv[ArrayBuilder[Type]] = unifyEnvsSeq(envs)

  def concatEnvs(envs: Seq[Env[ArrayBuilder[Type]]]): Env[ArrayBuilder[Type]] = {
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

  def unifyEnvsSeq(envs: Seq[BindingEnv[ArrayBuilder[Type]]]): BindingEnv[ArrayBuilder[Type]] = {
    val lc = envs.lengthCompare(1)
    if (lc < 0)
      BindingEnv.empty[ArrayBuilder[Type]]
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

  def memoizeTableIR(tir: TableIR, requestedType: TableType, memo: ComputeMutableState) {
    memo.requestedType.bind(tir, requestedType)
    tir match {
      case TableRead(_, _, _) =>
      case TableLiteral(_, _, _, _) =>
      case TableParallelize(rowsAndGlobal, _) =>
        memoizeValueIR(rowsAndGlobal, TStruct("rows" -> TArray(requestedType.rowType), "global" -> requestedType.globalType), memo)
      case TableRange(_, _) =>
      case TableRepartition(child, _, _) => memoizeTableIR(child, requestedType, memo)
      case TableHead(child, _) => memoizeTableIR(child, TableType(
        key = child.typ.key,
        rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, child.typ.key), requestedType.rowType),
        globalType = requestedType.globalType), memo)
      case TableTail(child, _) => memoizeTableIR(child, TableType(
        key = child.typ.key,
        rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, child.typ.key), requestedType.rowType),
        globalType = requestedType.globalType), memo)
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
        memoizeTableIR(left, leftDep, memo)

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
        memoizeTableIR(right, rightDep, memo)
      case TableLeftJoinRightDistinct(left, right, root) =>
        val fieldDep = requestedType.rowType.fieldOption(root).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val rightDep = TableType(
              key = right.typ.key,
              rowType = unify(
                right.typ.rowType,
                FastIndexedSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                  FastIndexedSeq(struct): _*),
              globalType = minimal(right.typ.globalType))
            memoizeTableIR(right, rightDep, memo)

            val lk = unifyKey(FastSeq(left.typ.key.take(right.typ.key.length), requestedType.key))
            val leftDep = TableType(
              key = lk,
              rowType = unify(left.typ.rowType, requestedType.rowType.filterSet(Set(root), include = false)._1,
                selectKey(left.typ.rowType, lk)),
              globalType = requestedType.globalType)
            memoizeTableIR(left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(left, requestedType, memo)
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
                FastIndexedSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                  FastIndexedSeq(struct): _*),
              globalType = minimal(right.typ.globalType))
            memoizeTableIR(right, rightDep, memo)

            val lk = unifyKey(FastSeq(left.typ.key.take(right.typ.key.length), requestedType.key))
            val leftDep = TableType(
              key = lk,
              rowType = unify(left.typ.rowType, requestedType.rowType.filterSet(Set(root), include = false)._1,
                selectKey(left.typ.rowType, lk)),
              globalType = requestedType.globalType)
            memoizeTableIR(left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(left, requestedType, memo)
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
        children.foreach(memoizeTableIR(_, dep, memo))
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
        memoizeTableIR(child, dep, memo)
      case TableFilter(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, requestedType, irDep), memo)
      case TableKeyBy(child, _, isSorted) =>
        val reqKey = requestedType.key
        val isPrefix = reqKey.zip(child.typ.key).forall { case (l, r) => l == r }
        val childReqKey = if (isSorted)
          child.typ.key
        else if (isPrefix)
          if  (reqKey.length <= child.typ.key.length) reqKey else child.typ.key
        else FastIndexedSeq()

        memoizeTableIR(child, TableType(
          key = childReqKey,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, childReqKey), requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableOrderBy(child, sortFields) =>
        val k = if (sortFields.forall(_.sortOrder == Ascending) && child.typ.key.startsWith(sortFields.map(_.field)))
          child.typ.key
        else
          FastIndexedSeq()
        memoizeTableIR(child, TableType(
          key = k,
          rowType = unify(child.typ.rowType,
            selectKey(child.typ.rowType, sortFields.map(_.field) ++ k),
            requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableDistinct(child) =>
        val dep = TableType(key = child.typ.key,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.key)),
          globalType = requestedType.globalType)
        memoizeTableIR(child, dep, memo)
      case TableMapPartitions(child, gName, pName, body) =>
        val reqRowsType = TStream(requestedType.rowType)
        val bodyDep = memoizeValueIR(body, reqRowsType, memo)
        val depGlobalType = unifySeq(child.typ.globalType,
          bodyDep.eval.lookupOption(gName).map(_.result()).getOrElse(Array()))
        val depRowType = unifySeq(child.typ.rowType,
          bodyDep.eval.lookupOption(pName)
            .map(_.result().map(_.asInstanceOf[TStream].elementType))
            .getOrElse(Array()))
        val dep = TableType(
          key = requestedType.key,
          rowType = depRowType.asInstanceOf[TStruct],
          globalType = depGlobalType.asInstanceOf[TStruct])
        memoizeTableIR(child, dep, memo)
      case TableMapRows(child, newRow) =>
        val rowDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        val dep = TableType(
          key = requestedType.key,
          rowType = unify(child.typ.rowType, selectKey(requestedType.rowType, requestedType.key), rowDep.rowType),
          globalType = unify(child.typ.globalType, requestedType.globalType, rowDep.globalType)
        )
        memoizeTableIR(child, dep, memo)
      case TableMapGlobals(child, newGlobals) =>
        val globalDep = memoizeAndGetDep(newGlobals, requestedType.globalType, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, requestedType.copy(globalType = globalDep.globalType), globalDep), memo)
      case TableAggregateByKey(child, newRow) =>
        val aggDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        memoizeTableIR(child, TableType(key = child.typ.key,
          rowType = unify(child.typ.rowType, aggDep.rowType, selectKey(child.typ.rowType, child.typ.key)),
          globalType = unify(child.typ.globalType, aggDep.globalType, requestedType.globalType)), memo)
      case TableKeyByAndAggregate(child, expr, newKey, _, _) =>
        val keyDep = memoizeAndGetDep(newKey, newKey.typ, child.typ, memo)
        val exprDep = memoizeAndGetDep(expr, requestedType.valueType, child.typ, memo)
        memoizeTableIR(child,
          TableType(
            key = FastIndexedSeq(), // note: this can deoptimize if prune runs before Simplify
            rowType = unify(child.typ.rowType, keyDep.rowType, exprDep.rowType),
            globalType = unify(child.typ.globalType, keyDep.globalType, exprDep.globalType, requestedType.globalType)),
          memo)
      case TableGroupWithinPartitions(child, name, n) =>
        memoizeTableIR(child, child.typ, memo)
      case MatrixColsTable(child) =>
        val mtDep = minimal(child.typ).copy(
          globalType = requestedType.globalType,
          entryType = TStruct.empty,
          colType = requestedType.rowType,
          colKey = requestedType.key)
        memoizeMatrixIR(child, mtDep, memo)
      case MatrixRowsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = requestedType.globalType,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, requestedType.key), requestedType.rowType),
          rowKey = requestedType.key)
        memoizeMatrixIR(child, mtDep, memo)
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
        memoizeMatrixIR(child, mtDep, memo)
      case TableUnion(children) =>
        children.foreach(memoizeTableIR(_, requestedType, memo))
      case CastMatrixToTable(child, entriesFieldName, colsFieldName) =>
        val childDep = MatrixType(
          rowKey = requestedType.key,
          colKey = FastIndexedSeq(),
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
        memoizeMatrixIR(child, childDep, memo)
      case TableRename(child, rowMap, globalMap) =>
        val rowMapRev = rowMap.map { case (k, v) => (v, k) }
        val globalMapRev = globalMap.map { case (k, v) => (v, k) }
        val childDep = TableType(
          rowType = requestedType.rowType.rename(rowMapRev),
          globalType = requestedType.globalType.rename(globalMapRev),
          key = requestedType.key.map(k => rowMapRev.getOrElse(k, k)))
        memoizeTableIR(child, childDep, memo)
      case TableFilterIntervals(child, _, _) =>
        memoizeTableIR(child, requestedType.copy(key = child.typ.key,
          rowType = PruneDeadFields.unify(child.typ.rowType,
            requestedType.rowType,
            PruneDeadFields.selectKey(child.typ.rowType, child.typ.key))), memo)
      case TableToTableApply(child, f) => memoizeTableIR(child, child.typ, memo)
      case MatrixToTableApply(child, _) => memoizeMatrixIR(child, child.typ, memo)
      case BlockMatrixToTableApply(bm, aux, _) =>
        memoizeBlockMatrixIR(bm, bm.typ, memo)
        memoizeValueIR(aux, aux.typ, memo)
      case BlockMatrixToTable(child) => memoizeBlockMatrixIR(child, child.typ, memo)
      case RelationalLetTable(name, value, body) =>
        memoizeTableIR(body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(value, unifySeq(value.typ, usages), memo)
    }
  }

  def memoizeMatrixIR(mir: MatrixIR, requestedType: MatrixType, memo: ComputeMutableState) {
    memo.requestedType.bind(mir, requestedType)
    mir match {
      case MatrixFilterCols(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterRows(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterEntries(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType, irDep), memo)
      case MatrixUnionCols(left, right, joinType) =>
        val leftRequestedType = requestedType.copy(
          rowKey = left.typ.rowKey,
          rowType = unify(left.typ.rowType, requestedType.rowType, selectKey(left.typ.rowType, left.typ.rowKey))
        )
        val rightRequestedType = requestedType.copy(
          globalType = TStruct.empty,
          rowKey = right.typ.rowKey,
          rowType = selectKey(right.typ.rowType, right.typ.rowKey))
        memoizeMatrixIR(left, leftRequestedType, memo)
        memoizeMatrixIR(right, rightRequestedType, memo)
      case MatrixMapEntries(child, newEntries) =>
        val irDep = memoizeAndGetDep(newEntries, requestedType.entryType, child.typ, memo)
        val depMod = requestedType.copy(entryType = TStruct.empty)
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixKeyRowsBy(child, _, isSorted) =>
        val reqKey = requestedType.rowKey
        val childReqKey = if (isSorted) child.typ.rowKey.take(reqKey.length) else FastIndexedSeq()
        memoizeMatrixIR(child, requestedType.copy(
          rowKey = childReqKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, childReqKey))),
          memo)
      case MatrixMapRows(child, newRow) =>
        val irDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        val depMod = requestedType.copy(rowType = selectKey(child.typ.rowType, child.typ.rowKey))
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapCols(child, newCol, newKey) =>
        val irDep = memoizeAndGetDep(newCol, requestedType.colType, child.typ, memo)
        val reqKey =  newKey match {
          case Some(_) => FastIndexedSeq()
          case None => requestedType.colKey
        }
        val depMod = requestedType.copy(colType = selectKey(child.typ.colType, reqKey), colKey = reqKey)
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapGlobals(child, newGlobals) =>
        val irDep = memoizeAndGetDep(newGlobals, requestedType.globalType, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType.copy(globalType = irDep.globalType), irDep), memo)
      case MatrixRead(_, _, _, _) =>
      case MatrixLiteral(_, _) =>
      case MatrixChooseCols(child, _) =>
        memoizeMatrixIR(child, unify(child.typ, requestedType), memo)
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
        memoizeMatrixIR(child, explodedDep, memo)
      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val irDepEntry = memoizeAndGetDep(entryExpr, requestedType.entryType, child.typ, memo)
        val irDepRow = memoizeAndGetDep(rowExpr, requestedType.rowValueStruct, child.typ, memo)
        val childDep = MatrixType(
          rowKey = child.typ.rowKey,
          colKey = requestedType.colKey,
          entryType = irDepEntry.entryType,
          rowType = unify(child.typ.rowType, selectKey(child.typ.rowType, child.typ.rowKey), irDepRow.rowType, irDepEntry.rowType),
          colType = unify(child.typ.colType, requestedType.colType, irDepEntry.colType, irDepRow.colType),
          globalType = unify(child.typ.globalType, requestedType.globalType, irDepEntry.globalType, irDepRow.globalType))
        memoizeMatrixIR(child, childDep, memo)
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val irDepEntry = memoizeAndGetDep(entryExpr, requestedType.entryType, child.typ, memo)
        val irDepCol = memoizeAndGetDep(colExpr, requestedType.colValueStruct, child.typ, memo)
        val childDep: MatrixType = MatrixType(
          rowKey = requestedType.rowKey,
          colKey = child.typ.colKey,
          colType = unify(child.typ.colType, irDepCol.colType, irDepEntry.colType, selectKey(child.typ.colType, child.typ.colKey)),
          globalType = unify(child.typ.globalType, requestedType.globalType, irDepEntry.globalType, irDepCol.globalType),
          rowType = unify(child.typ.rowType, irDepEntry.rowType, irDepCol.rowType, requestedType.rowType),
          entryType = irDepEntry.entryType)
        memoizeMatrixIR(child, childDep, memo)
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
            memoizeTableIR(table, tableDep, memo)

            val mk = unifyKey(FastSeq(child.typ.rowKey.take(tk.length), requestedType.rowKey))
            val matDep = requestedType.copy(
              rowKey = mk,
              rowType =
                unify(child.typ.rowType,
                  selectKey(child.typ.rowType, mk),
                  requestedType.rowType.filterSet(Set(root), include = false)._1))
            memoizeMatrixIR(child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(child, requestedType, memo)
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
            memoizeTableIR(table, tableDep, memo)

            val mk = unifyKey(FastSeq(child.typ.colKey.take(table.typ.key.length), requestedType.colKey))
            val matDep = requestedType.copy(
              colKey = mk,
              colType = unify(child.typ.colType, requestedType.colType.filterSet(Set(uid), include = false)._1,
                selectKey(child.typ.colType, mk)))
            memoizeMatrixIR(child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(child, requestedType, memo)
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
        memoizeMatrixIR(child, dep, memo)
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
        memoizeMatrixIR(child, dep, memo)
      case MatrixRepartition(child, _, _) =>
        memoizeMatrixIR(child, requestedType, memo)
      case MatrixUnionRows(children) =>
        children.foreach(memoizeMatrixIR(_, requestedType, memo))
      case MatrixDistinctByRow(child) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.rowKey))
        )
        memoizeMatrixIR(child, dep, memo)
      case MatrixRowsHead(child, n) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.rowKey))
        )
        memoizeMatrixIR(child, dep, memo)
      case MatrixColsHead(child, n) => memoizeMatrixIR(child, requestedType, memo)
      case MatrixRowsTail(child, n) =>
        val dep = requestedType.copy(
          rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType, requestedType.rowType, selectKey(child.typ.rowType, child.typ.rowKey))
        )
        memoizeMatrixIR(child, dep, memo)
      case MatrixColsTail(child, n) => memoizeMatrixIR(child, requestedType, memo)
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        val m = Map(MatrixType.entriesIdentifier -> entriesFieldName)
        val childDep = child.typ.copy(
          key = requestedType.rowKey,
          globalType = unify(child.typ.globalType, requestedType.globalType, TStruct((colsFieldName, TArray(requestedType.colType)))),
          rowType = unify(child.typ.rowType, requestedType.rowType, TStruct((entriesFieldName, TArray(requestedType.entryType))))
        )
        memoizeTableIR(child, childDep, memo)
      case MatrixFilterIntervals(child, _, _) =>
        memoizeMatrixIR(child, requestedType.copy(rowKey = child.typ.rowKey,
          rowType = unify(child.typ.rowType,
            requestedType.rowType,
            selectKey(child.typ.rowType, child.typ.rowKey))), memo)
      case MatrixToMatrixApply(child, f) => memoizeMatrixIR(child, child.typ, memo)
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
        memoizeMatrixIR(child, childDep, memo)
      case RelationalLetMatrixTable(name, value, body) =>
        memoizeMatrixIR(body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(value, unifySeq(value.typ, usages), memo)
    }
  }

  def memoizeBlockMatrixIR(bmir: BlockMatrixIR, requestedType: BlockMatrixType, memo: ComputeMutableState): Unit = {
    memo.requestedType.bind(bmir, requestedType)
    bmir match {
      case RelationalLetBlockMatrix(name, value, body) =>
        memoizeBlockMatrixIR(body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(value, unifySeq(value.typ, usages), memo)
      case _ =>
        bmir.children.foreach {
          case mir: MatrixIR => memoizeMatrixIR(mir, mir.typ, memo)
          case tir: TableIR => memoizeTableIR(tir, tir.typ, memo)
          case bmir: BlockMatrixIR => memoizeBlockMatrixIR(bmir, bmir.typ, memo)
          case ir: IR => memoizeValueIR(ir, ir.typ, memo)
        }
    }
  }

  def memoizeAndGetDep(ir: IR, requestedType: Type, base: TableType, memo: ComputeMutableState): TableType = {
    val depEnv = memoizeValueIR(ir, requestedType, memo)
    val depEnvUnified = concatEnvs(FastIndexedSeq(depEnv.eval) ++ FastIndexedSeq(depEnv.agg, depEnv.scan).flatten)

    val expectedBindingSet = Set("row", "global")
    depEnvUnified.m.keys.foreach { k =>
      if (!expectedBindingSet.contains(k))
        throw new RuntimeException(s"found unexpected free variable in pruning: $k\n" +
          s"  ${ depEnv.pretty(_.result().mkString(",")) }\n" +
          s"  ${ Pretty(ir) }")
    }

    val min = minimal(base)
    val rowType = unifySeq(base.rowType,
      Array(min.rowType) ++ depEnvUnified.lookupOption("row").map(_.result()).getOrElse(Array()))
    val globalType = unifySeq(base.globalType,
      Array(min.globalType) ++ depEnvUnified.lookupOption("global").map(_.result()).getOrElse(Array()))
    TableType(key = FastIndexedSeq(),
      rowType = rowType.asInstanceOf[TStruct],
      globalType = globalType.asInstanceOf[TStruct])
  }

  def memoizeAndGetDep(ir: IR, requestedType: Type, base: MatrixType, memo: ComputeMutableState): MatrixType = {
    val depEnv = memoizeValueIR(ir, requestedType, memo)
    val depEnvUnified = concatEnvs(FastIndexedSeq(depEnv.eval) ++ FastIndexedSeq(depEnv.agg, depEnv.scan).flatten)

    val expectedBindingSet = Set("va", "sa", "g", "global", "n_rows", "n_cols")
    depEnvUnified.m.keys.foreach { k =>
      if (!expectedBindingSet.contains(k))
        throw new RuntimeException(s"found unexpected free variable in pruning: $k\n  ${ Pretty(ir) }")
    }

    val min = minimal(base)
    val globalType = unifySeq(base.globalType,
      Array(min.globalType) ++ depEnvUnified.lookupOption("global").map(_.result()).getOrElse(Array()))
      .asInstanceOf[TStruct]
    val rowType = unifySeq(base.rowType,
      Array(min.rowType) ++ depEnvUnified.lookupOption("va").map(_.result()).getOrElse(Array()))
      .asInstanceOf[TStruct]
    val colType = unifySeq(base.colType,
      Array(min.colType) ++ depEnvUnified.lookupOption("sa").map(_.result()).getOrElse(Array()))
      .asInstanceOf[TStruct]
    val entryType = unifySeq(base.entryType,
      Array(min.entryType) ++ depEnvUnified.lookupOption("g").map(_.result()).getOrElse(Array()))
      .asInstanceOf[TStruct]

    if (rowType.hasField(MatrixType.entriesIdentifier))
      throw new RuntimeException(s"prune: found dependence on entry array in row binding:\n${ Pretty(ir) }")

    MatrixType(
      rowKey = FastIndexedSeq(),
      colKey = FastIndexedSeq(),
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
  def memoizeValueIR(ir: IR, requestedType: Type, memo: ComputeMutableState): BindingEnv[ArrayBuilder[Type]] = {
    memo.requestedType.bind(ir, requestedType)
    ir match {
      case IsNA(value) => memoizeValueIR(value, minimal(value.typ), memo)
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

        memoizeValueIR(v, recur(requestedType, _typ, v.typ), memo)
      case If(cond, cnsq, alt) =>
        unifyEnvs(
          memoizeValueIR(cond, cond.typ, memo),
          memoizeValueIR(cnsq, requestedType, memo),
          memoizeValueIR(alt, requestedType, memo)
        )
      case Coalesce(values) => unifyEnvsSeq(values.map(memoizeValueIR(_, requestedType, memo)))
      case Consume(value) => memoizeValueIR(value, value.typ, memo)
      case Let(name, value, body) =>
        val bodyEnv = memoizeValueIR(body, requestedType, memo)
        val valueType = bodyEnv.eval.lookupOption(name) match {
          case Some(ab) => unifySeq(value.typ, ab.result())
          case None => minimal(value.typ)
        }
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(value, valueType, memo)
        )
      case AggLet(name, value, body, isScan) =>
        val bodyEnv = memoizeValueIR(body, requestedType, memo)
        if (isScan) {
          val valueType = unifySeq(
            value.typ,
            bodyEnv.scanOrEmpty.lookupOption(name).map(_.result()).getOrElse(Array()))

          val valueEnv = memoizeValueIR(value, valueType, memo)
          unifyEnvs(
            bodyEnv.copy(scan = bodyEnv.scan.map(_.delete(name))),
            valueEnv.copy(eval = Env.empty, scan = Some(valueEnv.eval))
          )
        } else {
          val valueType = unifySeq(
            value.typ,
            bodyEnv.aggOrEmpty.lookupOption(name).map(_.result()).getOrElse(Array()))

          val valueEnv = memoizeValueIR(value, valueType, memo)
          unifyEnvs(
            bodyEnv.copy(agg = bodyEnv.agg.map(_.delete(name))),
            valueEnv.copy(eval = Env.empty, agg = Some(valueEnv.eval))
          )
        }
      case Ref(name, t) =>
        val ab = new ArrayBuilder[Type]()
        ab += requestedType
        BindingEnv.empty.bindEval(name -> ab)
      case RelationalLet(name, value, body) =>
        val e = memoizeValueIR(body, requestedType, memo)
        val usages = memo.relationalRefs.get(name).map(_.result()).getOrElse(Array())
        memoizeValueIR(value, unifySeq(value.typ, usages), memo)
        e
      case RelationalRef(name, _) =>
        memo.relationalRefs.getOrElseUpdate(name, new ArrayBuilder[Type]) += requestedType
        BindingEnv.empty
      case MakeArray(args, _) =>
        val eltType = requestedType.asInstanceOf[TArray].elementType
        unifyEnvsSeq(args.map(a => memoizeValueIR(a, eltType, memo)))
      case MakeStream(args, _) =>
        val eltType = requestedType.asInstanceOf[TStream].elementType
        unifyEnvsSeq(args.map(a => memoizeValueIR(a, eltType, memo)))
      case ArrayRef(a, i, s) =>
        unifyEnvs(
          memoizeValueIR(a, TArray(requestedType), memo),
          memoizeValueIR(i, i.typ, memo),
          memoizeValueIR(s, s.typ, memo)
        )
      case ArrayLen(a) =>
        memoizeValueIR(a, minimal(a.typ), memo)
      case StreamTake(a, len) =>
        unifyEnvs(
          memoizeValueIR(a, requestedType, memo),
          memoizeValueIR(len, len.typ, memo))
      case StreamDrop(a, len) =>
        unifyEnvs(
          memoizeValueIR(a, requestedType, memo),
          memoizeValueIR(len, len.typ, memo))
      case StreamMap(a, name, body) =>
        val aType = a.typ.asInstanceOf[TStream]
        val bodyEnv = memoizeValueIR(body,
          requestedType.asInstanceOf[TStream].elementType,
          memo)
        val valueType = unifySeq(
          aType.elementType,
          bodyEnv.eval.lookupOption(name).map(_.result()).getOrElse(Array()))
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(a, TStream(valueType), memo)
        )
      case StreamGrouped(a, size) =>
        unifyEnvs(
          memoizeValueIR(a, requestedType.asInstanceOf[TStream].elementType, memo),
          memoizeValueIR(size, size.typ, memo))
      case StreamGroupByKey(a, key) =>
        val reqStructT = coerce[TStruct](coerce[TStream](coerce[TStream](requestedType).elementType).elementType)
        val origStructT = coerce[TStruct](coerce[TStream](a.typ).elementType)
        memoizeValueIR(a, TStream(unify(origStructT, reqStructT, selectKey(origStructT, key))), memo)
      case StreamMerge(l, r, key) =>
        val reqStructT = coerce[TStruct](coerce[TStream](requestedType).elementType)
        val origStructT = coerce[TStruct](coerce[TStream](l.typ).elementType)
        val childReqT = TStream(unify(origStructT, reqStructT, selectKey(origStructT, key)))
        unifyEnvs(
          memoizeValueIR(l, childReqT, memo),
          memoizeValueIR(r, childReqT, memo))
      case StreamZip(as, names, body, behavior) =>
        val bodyEnv = memoizeValueIR(body,
          requestedType.asInstanceOf[TStream].elementType,
          memo)
        val valueTypes = names.zip(as).map { case (name, a) =>
          bodyEnv.eval.lookupOption(name).map(ab => unifySeq(coerce[TStream](a.typ).elementType, ab.result()))
        }
        if (behavior == ArrayZipBehavior.AssumeSameLength && valueTypes.forall(_.isEmpty)) {
          unifyEnvs(memoizeValueIR(as.head, TStream(minimal(coerce[TStream](as.head.typ).elementType)), memo) +:
                      Array(bodyEnv.deleteEval(names)): _*)
        } else {
          unifyEnvs(
            as.zip(valueTypes).map { case (a, vtOption) =>
              val at = coerce[TStream](a.typ)
              if (behavior == ArrayZipBehavior.AssumeSameLength) {
                vtOption.map { vt =>
                  memoizeValueIR(a, TStream(vt), memo)
                }.getOrElse(BindingEnv.empty)
              } else
                memoizeValueIR(a, TStream(vtOption.getOrElse(minimal(at.elementType))), memo)
            } ++ Array(bodyEnv.deleteEval(names)): _*)
        }
      case StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val eltType = coerce[TStruct](coerce[TStream](as.head.typ).elementType)
        val requestedEltType = coerce[TStream](requestedType).elementType
        val bodyEnv = memoizeValueIR(joinF, requestedEltType, memo)
        val childRequestedEltType = unifySeq(
          eltType,
          bodyEnv.eval.lookupOption(curVals).map(_.result().map(_.asInstanceOf[TArray].elementType)).getOrElse(Array()) :+
            selectKey(eltType, key))
        unifyEnvsSeq(as.map(memoizeValueIR(_, TStream(childRequestedEltType), memo)))
      case StreamMultiMerge(as, key) =>
        val eltType = coerce[TStruct](coerce[TStream](as.head.typ).elementType)
        val requestedEltType = coerce[TStream](requestedType).elementType
        val childRequestedEltType = unify(eltType, requestedEltType, selectKey(eltType, key))
        unifyEnvsSeq(as.map(memoizeValueIR(_, TStream(childRequestedEltType), memo)))
      case StreamFilter(a, name, cond) =>
        val aType = a.typ.asInstanceOf[TStream]
        val bodyEnv = memoizeValueIR(cond, cond.typ, memo)
        val valueType = unifySeq(
          aType.elementType,
          FastIndexedSeq(requestedType.asInstanceOf[TStream].elementType) ++
            bodyEnv.eval.lookupOption(name).map(_.result()).getOrElse(Array()))
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(a, TStream(valueType), memo)
        )
      case StreamFlatMap(a, name, body) =>
        val aType = a.typ.asInstanceOf[TStream]
        val bodyEnv = memoizeValueIR(body, requestedType, memo)
        val valueType = unifySeq(
          aType.elementType,
          bodyEnv.eval.lookupOption(name).map(_.result()).getOrElse(Array()))
        unifyEnvs(
          bodyEnv.deleteEval(name),
          memoizeValueIR(a, TStream(valueType), memo)
        )
      case StreamFold(a, zero, accumName, valueName, body) =>
        val aType = a.typ.asInstanceOf[TStream]
        val zeroEnv = memoizeValueIR(zero, zero.typ, memo)
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = unifySeq(
          aType.elementType,
          bodyEnv.eval.lookupOption(valueName).map(_.result()).getOrElse(Array()))

        unifyEnvs(
          zeroEnv,
          bodyEnv.deleteEval(valueName).deleteEval(accumName),
          memoizeValueIR(a, TStream(valueType), memo)
        )
      case StreamFold2(a, accum, valueName, seq, res) =>
        val aType = a.typ.asInstanceOf[TStream]
        val zeroEnvs = accum.map { case (name, zval) => memoizeValueIR(zval, zval.typ, memo) }
        val seqEnvs = seq.map { seq => memoizeValueIR(seq, seq.typ, memo) }
        val resEnv = memoizeValueIR(res, requestedType, memo)
        val valueType = unifySeq(
          aType.elementType,
          resEnv.eval.lookupOption(valueName).map(_.result()).getOrElse(Array()) ++
          seqEnvs.flatMap(_.eval.lookupOption(valueName).map(_.result()).getOrElse(Array())))

        val accumNames = accum.map(_._1)
        val seqNames = accumNames ++ Array(valueName)
        unifyEnvsSeq(
          zeroEnvs
            ++ Array(resEnv.copy(eval = resEnv.eval.delete(accumNames)))
            ++ seqEnvs.map(e => e.copy(eval = e.eval.delete(seqNames)))
            ++ Array(memoizeValueIR(a, TStream(valueType), memo))
        )
      case StreamScan(a, zero, accumName, valueName, body) =>
        val aType = a.typ.asInstanceOf[TStream]
        val zeroEnv = memoizeValueIR(zero, zero.typ, memo)
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = unifySeq(
          aType.elementType,
          bodyEnv.eval.lookupOption(valueName).map(_.result()).getOrElse(Array()))
        unifyEnvs(
          zeroEnv,
          bodyEnv.deleteEval(valueName).deleteEval(accumName),
          memoizeValueIR(a, TStream(valueType), memo)
        )
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val lType = left.typ.asInstanceOf[TStream]
        val rType = right.typ.asInstanceOf[TStream]

        val joinEnv = memoizeValueIR(join, requestedType.asInstanceOf[TStream].elementType, memo)

        val lRequested = unifySeq(
          lType.elementType,
          joinEnv.eval.lookupOption(l).map(_.result()).getOrElse(Array())
            :+ selectKey(lType.elementType.asInstanceOf[TStruct], lKey))

        val rRequested = unifySeq(
          rType.elementType,
          joinEnv.eval.lookupOption(r).map(_.result()).getOrElse(Array())
            :+ selectKey(rType.elementType.asInstanceOf[TStruct], rKey))

        unifyEnvs(
          joinEnv.deleteEval(l).deleteEval(r),
          memoizeValueIR(left, TStream(lRequested), memo),
          memoizeValueIR(right, TStream(rRequested), memo))
      case ArraySort(a, left, right, lessThan) =>
        val compEnv = memoizeValueIR(lessThan, lessThan.typ, memo)

        val aType = a.typ.asInstanceOf[TStream]
        val requestedElementType = unifySeq(
          aType.elementType,
          Array(requestedType.asInstanceOf[TArray].elementType) ++
            compEnv.eval.lookupOption(left).map(_.result()).getOrElse(Array()) ++
            compEnv.eval.lookupOption(right).map(_.result()).getOrElse(Array()))

        val aEnv = memoizeValueIR(a, TStream(requestedElementType), memo)

        unifyEnvs(
          compEnv.deleteEval(left).deleteEval(right),
          aEnv
        )
      case StreamFor(a, valueName, body) =>
        assert(requestedType == TVoid)
        val aType = a.typ.asInstanceOf[TStream]
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = unifySeq(
          aType.elementType,
          bodyEnv.eval.lookupOption(valueName).map(_.result()).getOrElse(Array()))
        unifyEnvs(
          bodyEnv.deleteEval(valueName),
          memoizeValueIR(a, TStream(valueType), memo)
        )
      case MakeNDArray(data, shape, rowMajor) =>
        val elementType = requestedType.asInstanceOf[TNDArray].elementType
        unifyEnvs(
          memoizeValueIR(data, TArray(elementType), memo),
          memoizeValueIR(shape, shape.typ, memo),
          memoizeValueIR(rowMajor, rowMajor.typ, memo)
        )
      case NDArrayMap(nd, valueName, body) =>
        val ndType = nd.typ.asInstanceOf[TNDArray]
        val bodyEnv = memoizeValueIR(body, requestedType.asInstanceOf[TNDArray].elementType, memo)
        val valueType = unifySeq(
          ndType.elementType,
          bodyEnv.eval.lookupOption(valueName).map(_.result()).getOrElse(Array())
        )
        unifyEnvs(
          bodyEnv.deleteEval(valueName),
          memoizeValueIR(nd, ndType.copy(elementType = valueType), memo)
        )
      case NDArrayMap2(left, right, leftName, rightName, body) =>
        val leftType = left.typ.asInstanceOf[TNDArray]
        val rightType = right.typ.asInstanceOf[TNDArray]
        val bodyEnv = memoizeValueIR(body, requestedType.asInstanceOf[TNDArray].elementType, memo)

        val leftValueType = unify(
          leftType.elementType,
          bodyEnv.eval.lookupOption(leftName).map(_.result()).getOrElse(Array()):_*
        )

        val rightValueType = unify(
          rightType.elementType,
          bodyEnv.eval.lookupOption(rightName).map(_.result()).getOrElse(Array()):_*
        )

        unifyEnvs(
          bodyEnv.deleteEval(leftName).deleteEval(rightName),
          memoizeValueIR(left, leftType.copy(elementType = leftValueType), memo),
          memoizeValueIR(right, rightType.copy(elementType = rightValueType), memo)
        )
      case AggExplode(a, name, body, isScan) =>
        val aType = a.typ.asInstanceOf[TStream]
        val bodyEnv = memoizeValueIR(body,
          requestedType,
          memo)
        if (isScan) {
          val valueType = unifySeq(
            aType.elementType,
            bodyEnv.scanOrEmpty.lookupOption(name).map(_.result()).getOrElse(Array()))

          val aEnv = memoizeValueIR(a, TStream(valueType), memo)
          unifyEnvs(
            BindingEnv(scan = bodyEnv.scan.map(_.delete(name))),
            BindingEnv(scan = Some(aEnv.eval))
          )
        } else {
          val valueType = unifySeq(
            aType.elementType,
            bodyEnv.aggOrEmpty.lookupOption(name).map(_.result()).getOrElse(Array()))

          val aEnv = memoizeValueIR(a, TStream(valueType), memo)
          unifyEnvs(
            BindingEnv(agg = bodyEnv.agg.map(_.delete(name))),
            BindingEnv(agg = Some(aEnv.eval))
          )
        }
      case AggFilter(cond, aggIR, isScan) =>
        val condEnv = memoizeValueIR(cond, cond.typ, memo)
        unifyEnvs(
          if (isScan)
            BindingEnv(scan = Some(condEnv.eval))
          else
            BindingEnv(agg = Some(condEnv.eval)),
          memoizeValueIR(aggIR, requestedType, memo)
        )
      case AggGroupBy(key, aggIR, isScan) =>
        val keyEnv = memoizeValueIR(key, requestedType.asInstanceOf[TDict].keyType, memo)
        unifyEnvs(
          if (isScan)
            BindingEnv(scan = Some(keyEnv.eval))
          else
            BindingEnv(agg = Some(keyEnv.eval)),
          memoizeValueIR(aggIR, requestedType.asInstanceOf[TDict].valueType, memo)
        )
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(aggBody,
          requestedType.asInstanceOf[TArray].elementType,
          memo)
        if (isScan) {
          val valueType = unifySeq(
            aType.elementType,
            bodyEnv.scanOrEmpty.lookupOption(elementName).map(_.result()).getOrElse(Array()))

          val aEnv = memoizeValueIR(a, TArray(valueType), memo)
          unifyEnvsSeq(FastSeq(
            bodyEnv.copy(eval = bodyEnv.eval.delete(indexName), scan = bodyEnv.scan.map(_.delete(elementName))),
            BindingEnv(scan = Some(aEnv.eval))
          ) ++ knownLength.map(x => memoizeValueIR(x, x.typ, memo)))
        } else {
          val valueType = unifySeq(
            aType.elementType,
            bodyEnv.aggOrEmpty.lookupOption(elementName).map(_.result()).getOrElse(Array()))

          val aEnv = memoizeValueIR(a, TArray(valueType), memo)
          unifyEnvsSeq(FastSeq(
            bodyEnv.copy(eval = bodyEnv.eval.delete(indexName), agg = bodyEnv.agg.map(_.delete(elementName))),
            BindingEnv(agg = Some(aEnv.eval))
          ) ++ knownLength.map(x => memoizeValueIR(x, x.typ, memo)))
        }
      case ApplyAggOp(initOpArgs, seqOpArgs, sig) =>
        val prunedSig = AggSignature.prune(sig, requestedType)
        val initEnv = unifyEnvsSeq(initOpArgs.zip(prunedSig.initOpArgs).map { case (arg, req) => memoizeValueIR(arg, req, memo) })
        val seqOpEnv = unifyEnvsSeq(seqOpArgs.zip(prunedSig.seqOpArgs).map { case (arg, req) => memoizeValueIR(arg, req, memo) })
        BindingEnv(eval = initEnv.eval, agg = Some(seqOpEnv.eval))
      case ApplyScanOp(initOpArgs, seqOpArgs, sig) =>
        val prunedSig = AggSignature.prune(sig, requestedType)
        val initEnv = unifyEnvsSeq(initOpArgs.zip(prunedSig.initOpArgs).map { case (arg, req) => memoizeValueIR(arg, req, memo) })
        val seqOpEnv = unifyEnvsSeq(seqOpArgs.zip(prunedSig.seqOpArgs).map { case (arg, req) => memoizeValueIR(arg, req, memo) })
        BindingEnv(eval = initEnv.eval, scan = Some(seqOpEnv.eval))
      case StreamAgg(a, name, query) =>
        val aType = a.typ.asInstanceOf[TStream]
        val queryEnv = memoizeValueIR(query, requestedType, memo)
        val requestedElemType = unifySeq(
          aType.elementType,
          queryEnv.aggOrEmpty.lookupOption(name).map(_.result()).getOrElse(Array()))
        val aEnv = memoizeValueIR(a, TStream(requestedElemType), memo)
        unifyEnvs(
          BindingEnv(eval = concatEnvs(Array(queryEnv.eval, queryEnv.aggOrEmpty.delete(name)))),
          aEnv)
      case StreamAggScan(a, name, query) =>
        val aType = a.typ.asInstanceOf[TStream]
        val queryEnv = memoizeValueIR(query, requestedType.asInstanceOf[TStream].elementType, memo)
        val requestedElemType = unifySeq(
          aType.elementType,
          queryEnv.scanOrEmpty.lookupOption(name).map(_.result()).getOrElse(Array()) ++
            queryEnv.eval.lookupOption(name).map(_.result()).getOrElse(Array()))
        val aEnv = memoizeValueIR(a, TStream(requestedElemType), memo)
        unifyEnvs(
          BindingEnv(eval = concatEnvs(Array(queryEnv.eval.delete(name), queryEnv.scanOrEmpty.delete(name)))),
          aEnv)
      case RunAgg(body, result, _) =>
        unifyEnvs(
          memoizeValueIR(body, body.typ, memo),
          memoizeValueIR(result, requestedType, memo)
        )
      case RunAggScan(array, name, init, seqs, result, signature) =>
        val aType = array.typ.asInstanceOf[TStream]
        val resultEnv = memoizeValueIR(result, requestedType.asInstanceOf[TStream].elementType, memo)
        val seqEnv = memoizeValueIR(seqs, seqs.typ, memo)
        val elemEnv = unifyEnvs(resultEnv, seqEnv)
        val requestedElemType = unifySeq(aType.elementType,
          elemEnv.eval.lookupOption(name).map(_.result()).getOrElse(Array()))
        unifyEnvs(
          elemEnv,
          memoizeValueIR(array, TStream(requestedElemType), memo),
          memoizeValueIR(init, init.typ, memo)
        )
      case MakeStruct(fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        unifyEnvsSeq(fields.flatMap { case (fname, fir) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          sType.fieldOption(fname).map(f => memoizeValueIR(fir, f.typ, memo))
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
          FastSeq(memoizeValueIR(old, leftDep, memo)) ++
            // ignore unreachable fields, these are eliminated on the upwards pass
            fields.flatMap { case (fname, fir) =>
              rightDep.fieldOption(fname).map(f => memoizeValueIR(fir, f.typ, memo))
            }
        )
      case SelectFields(old, fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        memoizeValueIR(old, TStruct(fields.flatMap(f => sType.fieldOption(f).map(f -> _.typ)): _*), memo)
      case GetField(o, name) =>
        memoizeValueIR(o, TStruct(name -> requestedType), memo)
      case MakeTuple(fields) =>
        val tType = requestedType.asInstanceOf[TTuple]

        unifyEnvsSeq(
          fields.flatMap { case (i, value) =>
            // ignore unreachable fields, these are eliminated on the upwards pass
            tType.fieldIndex.get(i)
              .map { idx =>
                memoizeValueIR(value, tType.types(idx), memo)
              }})
      case GetTupleElement(o, idx) =>
        val childTupleType = o.typ.asInstanceOf[TTuple]
        val tupleDep = TTuple(FastIndexedSeq(TupleField(idx, requestedType)))
        memoizeValueIR(o, tupleDep, memo)
      case MatrixCount(child) =>
        memoizeMatrixIR(child, minimal(child.typ), memo)
        BindingEnv.empty
      case TableCount(child) =>
        memoizeTableIR(child, minimal(child.typ), memo)
        BindingEnv.empty
      case TableGetGlobals(child) =>
        memoizeTableIR(child, minimal(child.typ).copy(globalType = requestedType.asInstanceOf[TStruct]), memo)
        BindingEnv.empty
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        memoizeTableIR(child, TableType(
          key = child.typ.key,
          rowType = unify(child.typ.rowType,
            rStruct.fieldOption("rows").map(_.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]).getOrElse(TStruct.empty)),
          globalType = rStruct.fieldOption("global").map(_.typ.asInstanceOf[TStruct]).getOrElse(TStruct.empty)),
          memo)
        BindingEnv.empty
      case TableToValueApply(child, _) =>
        memoizeTableIR(child, child.typ, memo)
        BindingEnv.empty
      case MatrixToValueApply(child, _) => memoizeMatrixIR(child, child.typ, memo)
        BindingEnv.empty
      case BlockMatrixToValueApply(child, _) => memoizeBlockMatrixIR(child, child.typ, memo)
        BindingEnv.empty
      case TableAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(query, query.typ, child.typ, memo)
        val dep = TableType(
          key = child.typ.key,
          rowType = unify(child.typ.rowType, queryDep.rowType, selectKey(child.typ.rowType, child.typ.key)),
          globalType = queryDep.globalType
        )
        memoizeTableIR(child, dep, memo)
        BindingEnv.empty
      case MatrixAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(query, query.typ, child.typ, memo)
        val dep = MatrixType(
          rowKey = child.typ.rowKey,
          colKey = FastIndexedSeq(),
          rowType = unify(child.typ.rowType, queryDep.rowType, selectKey(child.typ.rowType, child.typ.rowKey)),
          entryType = queryDep.entryType,
          colType = queryDep.colType,
          globalType = queryDep.globalType
        )
        memoizeMatrixIR(child, dep, memo)
        BindingEnv.empty
      case CollectDistributedArray(contexts, globals, cname, gname, body) =>
        val rArray = requestedType.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(body, rArray.elementType, memo)
        assert(bodyEnv.scan.isEmpty)
        assert(bodyEnv.agg.isEmpty)

        val cDep = bodyEnv.eval.lookupOption(cname) match {
          case Some(ts) => TStream(unify[Type](coerce[TStream](contexts.typ).elementType, ts.result(): _*))
          case None => minimal(contexts.typ)
        }

        val gDep = bodyEnv.eval.lookupOption(gname) match {
          case Some(ts) => unify[Type](globals.typ, ts.result(): _*)
          case None => minimal(globals.typ)
        }

        unifyEnvs(
          memoizeValueIR(contexts, cDep, memo),
          memoizeValueIR(globals, gDep, memo)
        )
      case _: IR =>
        val envs = ir.children.flatMap {
          case mir: MatrixIR =>
            memoizeMatrixIR(mir, mir.typ, memo)
            None
          case tir: TableIR =>
            memoizeTableIR(tir, tir.typ, memo)
            None
          case bmir: BlockMatrixIR => //NOTE Currently no BlockMatrixIRs would have dead fields
            None
          case ir: IR =>
            Some(memoizeValueIR(ir, ir.typ, memo))
        }
        unifyEnvsSeq(envs)
    }
  }

  def rebuild(tir: TableIR, memo: RebuildMutableState): TableIR = {
    val requestedType = memo.requestedType.lookup(tir).asInstanceOf[TableType]
    tir match {
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        TableParallelize(
          upcast(rebuildIR(rowsAndGlobal, BindingEnv.empty, memo),
            memo.requestedType.lookup(rowsAndGlobal).asInstanceOf[TStruct]),
          nPartitions)
      case TableRead(typ, dropRows, tr) =>
        // FIXME: remove this when all readers know how to read without keys
        val requestedTypeWithKey = TableType(
          key = typ.key,
          rowType = unify(typ.rowType, selectKey(typ.rowType, typ.key), requestedType.rowType),
          globalType = requestedType.globalType)
        TableRead(requestedTypeWithKey, dropRows, tr)
      case TableFilter(child, pred) =>
        val child2 = rebuild(child, memo)
        val pred2 = rebuildIR(pred, BindingEnv(child2.typ.rowEnv), memo)
        TableFilter(child2, pred2)
      case TableMapPartitions(child, gName, pName, body) =>
        val child2 = rebuild(child, memo)
        val body2 = rebuildIR(body, BindingEnv(Env(
          gName -> child2.typ.globalType,
          pName -> TStream(child2.typ.rowType))), memo)
        TableMapPartitions(child2, gName, pName, body2)
      case TableMapRows(child, newRow) =>
        val child2 = rebuild(child, memo)
        val newRow2 = rebuildIR(newRow, BindingEnv(child2.typ.rowEnv, scan = Some(child2.typ.rowEnv)), memo)
        val newRowType = newRow2.typ.asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.key.exists(k => !newRowType.hasField(k)))
          TableKeyBy(child2, child2.typ.key.takeWhile(newRowType.hasField))
        else
          child2
        TableMapRows(child2Keyed, newRow2)
      case TableMapGlobals(child, newGlobals) =>
        val child2 = rebuild(child, memo)
        TableMapGlobals(child2, rebuildIR(newGlobals, BindingEnv(child2.typ.globalEnv), memo))
      case TableKeyBy(child, _, isSorted) =>
        var child2 = rebuild(child, memo)
        val keys2 = requestedType.key
        // fully upcast before shuffle
        if (!isSorted && keys2.nonEmpty)
          child2 = upcastTable(child2, memo.requestedType.lookup(child).asInstanceOf[TableType], upcastGlobals = false)
        TableKeyBy(child2, keys2, isSorted)
      case TableOrderBy(child, sortFields) =>
        // fully upcast before shuffle
        val child2 = upcastTable(rebuild(child, memo), memo.requestedType.lookup(child).asInstanceOf[TableType], upcastGlobals = false)
        TableOrderBy(child2, sortFields)
      case TableLeftJoinRightDistinct(left, right, root) =>
        if (requestedType.rowType.hasField(root))
          TableLeftJoinRightDistinct(rebuild(left, memo), rebuild(right, memo), root)
        else
          rebuild(left, memo)
      case TableIntervalJoin(left, right, root, product) =>
        if (requestedType.rowType.hasField(root))
          TableIntervalJoin(rebuild(left, memo), rebuild(right, memo), root, product)
        else
          rebuild(left, memo)
      case TableMultiWayZipJoin(children, fieldName, globalName) =>
        val rebuilt = children.map { c => rebuild(c, memo) }
        val upcasted = rebuilt.map { t => upcastTable(t, memo.requestedType.lookup(children(0)).asInstanceOf[TableType]) }
        TableMultiWayZipJoin(upcasted, fieldName, globalName)
      case TableAggregateByKey(child, expr) =>
        val child2 = rebuild(child, memo)
        TableAggregateByKey(child2, rebuildIR(expr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo))
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val child2 = rebuild(child, memo)
        val expr2 = rebuildIR(expr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo)
        val newKey2 = rebuildIR(newKey, BindingEnv(child2.typ.rowEnv), memo)
        TableKeyByAndAggregate(child2, expr2, newKey2, nPartitions, bufferSize)
      case TableRename(child, rowMap, globalMap) =>
        val child2 = rebuild(child, memo)
        TableRename(
          child2,
          rowMap.filterKeys(child2.typ.rowType.hasField),
          globalMap.filterKeys(child2.typ.globalType.hasField))
      case TableUnion(children) =>
        val requestedType = memo.requestedType.lookup(tir).asInstanceOf[TableType]
        val rebuilt = children.map { c =>
          upcastTable(rebuild(c, memo), requestedType, upcastGlobals = false)
        }
        TableUnion(rebuilt)
      case RelationalLetTable(name, value, body) =>
        val value2 = rebuildIR(value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLetTable(name, value2, rebuild(body, memo))
      case BlockMatrixToTableApply(bmir, aux, function) =>
        val bmir2 = rebuild(bmir, memo)
        val aux2 = rebuildIR(aux, BindingEnv.empty, memo)
        BlockMatrixToTableApply(bmir2, aux2, function)
      case _ => tir.copy(tir.children.map {
        // IR should be a match error - all nodes with child value IRs should have a rule
        case childT: TableIR => rebuild(childT, memo)
        case childM: MatrixIR => rebuild(childM, memo)
        case childBm: BlockMatrixIR => rebuild(childBm, memo)
      })
    }
  }

  def rebuild(mir: MatrixIR, memo: RebuildMutableState): MatrixIR = {
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
        val child2 = rebuild(child, memo)
        MatrixFilterCols(child2, rebuildIR(pred, BindingEnv(child2.typ.colEnv), memo))
      case MatrixFilterRows(child, pred) =>
        val child2 = rebuild(child, memo)
        MatrixFilterRows(child2, rebuildIR(pred, BindingEnv(child2.typ.rowEnv), memo))
      case MatrixFilterEntries(child, pred) =>
        val child2 = rebuild(child, memo)
        MatrixFilterEntries(child2, rebuildIR(pred, BindingEnv(child2.typ.entryEnv), memo))
      case MatrixMapEntries(child, newEntries) =>
        val child2 = rebuild(child, memo)
        MatrixMapEntries(child2, rebuildIR(newEntries, BindingEnv(child2.typ.entryEnv), memo))
      case MatrixMapRows(child, newRow) =>
        val child2 = rebuild(child, memo)
        val newRow2 = rebuildIR(newRow,
          BindingEnv(child2.typ.rowEnv, agg = Some(child2.typ.entryEnv), scan = Some(child2.typ.rowEnv)), memo)
        val newRowType = newRow2.typ.asInstanceOf[TStruct]
        val child2Keyed = if (child2.typ.rowKey.exists(k => !newRowType.hasField(k)))
          MatrixKeyRowsBy(child2, child2.typ.rowKey.takeWhile(newRowType.hasField))
        else
          child2
        MatrixMapRows(child2Keyed, newRow2)
      case MatrixMapCols(child, newCol, newKey) =>
        val child2 = rebuild(child, memo)
        val newCol2 = rebuildIR(newCol,
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
        val child2 = rebuild(child, memo)
        MatrixMapGlobals(child2, rebuildIR(newGlobals, BindingEnv(child2.typ.globalEnv), memo))
      case MatrixKeyRowsBy(child, keys, isSorted) =>
        val child2 = rebuild(child, memo)
        val keys2 = keys.takeWhile(child2.typ.rowType.hasField)
        MatrixKeyRowsBy(child2, keys2, isSorted)
      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val child2 = rebuild(child, memo)
        MatrixAggregateRowsByKey(child2,
          rebuildIR(entryExpr, BindingEnv(child2.typ.colEnv, agg = Some(child2.typ.entryEnv)), memo),
          rebuildIR(rowExpr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo))
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val child2 = rebuild(child, memo)
        MatrixAggregateColsByKey(child2,
          rebuildIR(entryExpr, BindingEnv(child2.typ.rowEnv, agg = Some(child2.typ.entryEnv)), memo),
          rebuildIR(colExpr, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.colEnv)), memo))
      case MatrixUnionRows(children) =>
        val requestedType = memo.requestedType.lookup(mir).asInstanceOf[MatrixType]
        MatrixUnionRows(children.map { child =>
          upcast(rebuild(child, memo), requestedType,
            upcastGlobals = false)
        })
      case MatrixAnnotateRowsTable(child, table, root, product) =>
        // if the field is not used, this node can be elided entirely
        if (!requestedType.rowType.hasField(root))
          rebuild(child, memo)
        else {
          val child2 = rebuild(child, memo)
          val table2 = rebuild(table, memo)
          MatrixAnnotateRowsTable(child2, table2, root, product)
        }
      case MatrixAnnotateColsTable(child, table, uid) =>
        // if the field is not used, this node can be elided entirely
        if (!requestedType.colType.hasField(uid))
          rebuild(child, memo)
        else {
          val child2 = rebuild(child, memo)
          val table2 = rebuild(table, memo)
          MatrixAnnotateColsTable(child2, table2, uid)
        }
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        val child2 = rebuild(child, memo)
        MatrixRename(
          child2,
          globalMap.filterKeys(child2.typ.globalType.hasField),
          colMap.filterKeys(child2.typ.colType.hasField),
          rowMap.filterKeys(child2.typ.rowType.hasField),
          entryMap.filterKeys(child2.typ.entryType.hasField))
      case RelationalLetMatrixTable(name, value, body) =>
        val value2 = rebuildIR(value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLetMatrixTable(name, value2, rebuild(body, memo))
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        CastTableToMatrix(rebuild(child, memo), entriesFieldName, colsFieldName, requestedType.colKey)
      case _ => mir.copy(mir.children.map {
        // IR should be a match error - all nodes with child value IRs should have a rule
        case childT: TableIR => rebuild(childT, memo)
        case childM: MatrixIR => rebuild(childM, memo)
      })
    }
  }

  def rebuild(bmir: BlockMatrixIR, memo: RebuildMutableState): BlockMatrixIR = bmir match {
    case RelationalLetBlockMatrix(name, value, body) =>
      val value2 = rebuildIR(value, BindingEnv.empty, memo)
      memo.relationalRefs += name -> value2.typ
      RelationalLetBlockMatrix(name, value2, rebuild(body, memo))
    case _ =>
      bmir.copy(
        bmir.children.map {
          case tir: TableIR => rebuild(tir, memo)
          case mir: MatrixIR => rebuild(mir, memo)
          case ir: IR => rebuildIR(ir, BindingEnv.empty[Type], memo)
          case bmir: BlockMatrixIR => rebuild(bmir, memo)
        }
      )
  }

  def rebuildIR(ir: IR, env: BindingEnv[Type], memo: RebuildMutableState): IR = {
    val requestedType = memo.requestedType.lookup(ir).asInstanceOf[Type]
    ir match {
      case NA(_) => NA(requestedType)
      case CastRename(v, _typ) =>
        val v2 = rebuildIR(v, env, memo)

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
        val cond2 = rebuildIR(cond, env, memo)
        val cnsq2 = rebuildIR(cnsq, env, memo)
        val alt2 = rebuildIR(alt, env, memo)

        if (cnsq2.typ == alt2.typ)
          If(cond2, cnsq2, alt2)
        else
          If(cond2,
            upcast(cnsq2, requestedType),
            upcast(alt2, requestedType)
          )
      case Coalesce(values) =>
        val values2 = values.map(rebuildIR(_, env, memo))
        require(values2.nonEmpty)
        if (values2.forall(_.typ == values2.head.typ))
          Coalesce(values2)
        else
          Coalesce(values2.map(upcast(_, requestedType)))
      case Consume(value) =>
        val value2 = rebuildIR(value, env, memo)
        Consume(value2)
      case Let(name, value, body) =>
        val value2 = rebuildIR(value, env, memo)
        Let(
          name,
          value2,
          rebuildIR(body, env.bindEval(name, value2.typ), memo)
        )
      case AggLet(name, value, body, isScan) =>
        val value2 = rebuildIR(value, if (isScan) env.promoteScan else env.promoteAgg, memo)
        AggLet(
          name,
          value2,
          rebuildIR(body, if (isScan) env.bindScan(name, value2.typ) else env.bindAgg(name, value2.typ), memo),
          isScan
        )
      case Ref(name, t) =>
        Ref(name, env.eval.lookupOption(name).getOrElse(t))
      case RelationalLet(name, value, body) =>
        val value2 = rebuildIR(value, BindingEnv.empty, memo)
        memo.relationalRefs += name -> value2.typ
        RelationalLet(name, value2, rebuildIR(body, env, memo))
      case RelationalRef(name, _) => RelationalRef(name, memo.relationalRefs(name))
      case MakeArray(args, _) =>
        val dep = requestedType.asInstanceOf[TArray]
        val args2 = args.map(a => rebuildIR(a, env, memo))
        MakeArray.unify(args2, TArray(dep.elementType))
      case MakeStream(args, _) =>
        val dep = requestedType.asInstanceOf[TStream]
        val args2 = args.map(a => rebuildIR(a, env, memo))
        MakeStream.unify(args2, TStream(dep.elementType))
      case StreamMap(a, name, body) =>
        val a2 = rebuildIR(a, env, memo)
        StreamMap(a2, name, rebuildIR(body, env.bindEval(name, a2.typ.asInstanceOf[TStream].elementType), memo))
      case StreamMerge(l, r, key) =>
        val l2 = rebuildIR(l, env, memo)
        val r2 = rebuildIR(r, env, memo)
        if (l2.typ == r2.typ)
          StreamMerge(l2, r2, key)
        else
          StreamMerge(
            upcast(l2, memo.requestedType.lookup(l).asInstanceOf[Type]),
            upcast(r2, memo.requestedType.lookup(r).asInstanceOf[Type]),
            key)
      case StreamZip(as, names, body, b) =>
        val (newAs, newNames) = as.zip(names)
          .flatMap { case (a, name) => if (memo.requestedType.contains(a)) Some((rebuildIR(a, env, memo), name)) else None }
          .unzip
        StreamZip(newAs, newNames, rebuildIR(body,
          env.bindEval(newNames.zip(newAs.map(a => a.typ.asInstanceOf[TStream].elementType)): _*), memo), b)
      case StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val newAs = as.map(a => rebuildIR(a, env, memo))
        val newEltType = as.head.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
        val newJoinF = rebuildIR(
          joinF,
          env.bindEval(curKey -> selectKey(newEltType, key), curVals -> TArray(newEltType)),
          memo)
        StreamZipJoin(newAs, key, curKey, curVals, newJoinF)
      case StreamFilter(a, name, cond) =>
        val a2 = rebuildIR(a, env, memo)
        StreamFilter(a2, name, rebuildIR(cond, env.bindEval(name, a2.typ.asInstanceOf[TStream].elementType), memo))
      case StreamFlatMap(a, name, body) =>
        val a2 = rebuildIR(a, env, memo)
        StreamFlatMap(a2, name, rebuildIR(body, env.bindEval(name, a2.typ.asInstanceOf[TStream].elementType), memo))
      case StreamFold(a, zero, accumName, valueName, body) =>
        val a2 = rebuildIR(a, env, memo)
        val z2 = rebuildIR(zero, env, memo)
        StreamFold(
          a2,
          z2,
          accumName,
          valueName,
          rebuildIR(body, env.bindEval(accumName -> z2.typ, valueName -> a2.typ.asInstanceOf[TStream].elementType), memo)
        )
      case StreamFold2(a: IR, accum, valueName, seqs, result) =>
        val a2 = rebuildIR(a, env, memo)
        val newAccum = accum.map { case (n, z) => n -> rebuildIR(z, env, memo) }
        val newEnv = env
          .bindEval(valueName -> a2.typ.asInstanceOf[TStream].elementType)
          .bindEval(newAccum.map { case (n, z) => n -> z.typ }: _*)
        StreamFold2(
          a2,
          newAccum,
          valueName,
          seqs.map(rebuildIR(_, newEnv, memo)),
          rebuildIR(result, newEnv, memo))
      case StreamScan(a, zero, accumName, valueName, body) =>
        val a2 = rebuildIR(a, env, memo)
        val z2 = rebuildIR(zero, env, memo)
        StreamScan(
          a2,
          z2,
          accumName,
          valueName,
          rebuildIR(body, env.bindEval(accumName -> z2.typ, valueName -> a2.typ.asInstanceOf[TStream].elementType), memo)
        )
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val left2 = rebuildIR(left, env, memo)
        val right2 = rebuildIR(right, env, memo)

        val ltyp = left2.typ.asInstanceOf[TStream]
        val rtyp = right2.typ.asInstanceOf[TStream]
        StreamJoinRightDistinct(
          left2, right2, lKey, rKey, l, r,
          rebuildIR(join, env.bindEval(l -> ltyp.elementType, r -> rtyp.elementType), memo),
          joinType)
      case StreamFor(a, valueName, body) =>
        val a2 = rebuildIR(a, env, memo)
        val body2 = rebuildIR(body, env.bindEval(valueName -> a2.typ.asInstanceOf[TStream].elementType), memo)
        StreamFor(a2, valueName, body2)
      case ArraySort(a, left, right, lessThan) =>
        val a2 = rebuildIR(a, env, memo)
        val et = a2.typ.asInstanceOf[TStream].elementType
        val lessThan2 = rebuildIR(lessThan, env.bindEval(left -> et, right -> et), memo)
        ArraySort(a2, left, right, lessThan2)
      case MakeNDArray(data, shape, rowMajor) =>
        val data2 = rebuildIR(data, env, memo)
        val shape2 = rebuildIR(shape, env, memo)
        val rowMajor2 = rebuildIR(rowMajor, env, memo)
        MakeNDArray(data2, shape2, rowMajor2)
      case NDArrayMap(nd, valueName, body) =>
        val nd2 = rebuildIR(nd, env, memo)
        NDArrayMap(nd2, valueName, rebuildIR(body, env.bindEval(valueName, nd2.typ.asInstanceOf[TNDArray].elementType), memo))
      case NDArrayMap2(left, right, leftName, rightName, body) =>
        val left2 = rebuildIR(left, env, memo)
        val right2 = rebuildIR(right, env, memo)
        val body2 = rebuildIR(body,
          env.bindEval(leftName, left2.typ.asInstanceOf[TNDArray].elementType).bindEval(rightName, right2.typ.asInstanceOf[TNDArray].elementType),
          memo)
        NDArrayMap2(left2, right2, leftName, rightName, body2)
      case MakeStruct(fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        // drop unnecessary field IRs
        val depFields = depStruct.fieldNames.toSet
        MakeStruct(fields.flatMap { case (f, fir) =>
          if (depFields.contains(f))
            Some(f -> rebuildIR(fir, env, memo))
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
            Some(i -> rebuildIR(f, env, memo))
          else
            None
        })
      case InsertFields(old, fields, fieldOrder) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val depFields = depStruct.fieldNames.toSet
        val rebuiltChild = rebuildIR(old, env, memo)
        val preservedChildFields = rebuiltChild.typ.asInstanceOf[TStruct].fieldNames.toSet
        InsertFields(rebuiltChild,
          fields.flatMap { case (f, fir) =>
            if (depFields.contains(f))
              Some(f -> rebuildIR(fir, env, memo))
            else {
              log.info(s"Prune: InsertFields: eliminating field '$f'")
              None
            }
          }, fieldOrder.map(fds => fds.filter(f => depFields.contains(f) || preservedChildFields.contains(f))))
      case SelectFields(old, fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val old2 = rebuildIR(old, env, memo)
        SelectFields(old2, fields.filter(f => old2.typ.asInstanceOf[TStruct].hasField(f) && depStruct.hasField(f)))
      case TableAggregate(child, query) =>
        val child2 = rebuild(child, memo)
        val query2 = rebuildIR(query, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.rowEnv)), memo)
        TableAggregate(child2, query2)
      case MatrixAggregate(child, query) =>
        val child2 = rebuild(child, memo)
        val query2 = rebuildIR(query, BindingEnv(child2.typ.globalEnv, agg = Some(child2.typ.entryEnv)), memo)
        MatrixAggregate(child2, query2)
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        if (!rStruct.hasField("rows"))
          if (rStruct.hasField("global"))
            MakeStruct(FastSeq("global" -> TableGetGlobals(rebuild(child, memo))))
          else
            MakeStruct(FastSeq())
        else
          TableCollect(rebuild(child, memo))
      case AggExplode(array, name, aggBody, isScan) =>
        val a2 = rebuildIR(array, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val a2t = a2.typ.asInstanceOf[TStream].elementType
        val body2 = rebuildIR(aggBody, if (isScan) env.bindScan(name, a2t) else env.bindAgg(name, a2t), memo)
        AggExplode(a2, name, body2, isScan)
      case AggFilter(cond, aggIR, isScan) =>
        val cond2 = rebuildIR(cond, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val aggIR2 = rebuildIR(aggIR, env, memo)
        AggFilter(cond2, aggIR2, isScan)
      case AggGroupBy(key, aggIR, isScan) =>
        val key2 = rebuildIR(key, if (isScan) env.promoteScan else env.promoteAgg, memo)
        val aggIR2 = rebuildIR(aggIR, env, memo)
        AggGroupBy(key2, aggIR2, isScan)
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) =>
        val aEnv = if (isScan) env.promoteScan else env.promoteAgg
        val a2 = rebuildIR(a, aEnv, memo)
        val a2t = a2.typ.asInstanceOf[TArray].elementType
        val env_ = env.bindEval(indexName -> TInt32)
        val aggBody2 = rebuildIR(aggBody, if (isScan) env_.bindScan(elementName, a2t) else env_.bindAgg(elementName, a2t), memo)
        AggArrayPerElement(a2, elementName, indexName, aggBody2, knownLength.map(rebuildIR(_, aEnv, memo)), isScan)
      case StreamAgg(a, name, query) =>
        val a2 = rebuildIR(a, env, memo)
        val query2 = rebuildIR(query, env.copy(agg = Some(env.eval.bind(name -> a2.typ.asInstanceOf[TStream].elementType))), memo)
        StreamAgg(a2, name, query2)
      case StreamAggScan(a, name, query) =>
        val a2 = rebuildIR(a, env, memo)
        val query2 = rebuildIR(query, env.copy(scan = Some(env.eval.bind(name -> a2.typ.asInstanceOf[TStream].elementType))), memo)
        StreamAggScan(a2, name, query2)
      case RunAgg(body, result, signatures) =>
        val body2 = rebuildIR(body, env, memo)
        val result2 = rebuildIR(result, env, memo)
        RunAgg(body2, result2, signatures)
      case RunAggScan(array, name, init, seqs, result, signature) =>
        val array2 = rebuildIR(array, env, memo)
        val init2 = rebuildIR(init, env, memo)
        val eltEnv = env.bindEval(name, array2.typ.asInstanceOf[TStream].elementType)
        val seqs2 = rebuildIR(seqs, eltEnv, memo)
        val result2 = rebuildIR(result, eltEnv, memo)
        RunAggScan(array2, name, init2, seqs2, result2, signature)
      case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
        val initOpArgs2 = initOpArgs.map(rebuildIR(_, env, memo))
        val seqOpArgs2 = seqOpArgs.map(rebuildIR(_, env.promoteAgg, memo))
        ApplyAggOp(initOpArgs2, seqOpArgs2,
          aggSig.copy(
            initOpArgs = initOpArgs2.map(_.typ),
            seqOpArgs = seqOpArgs2.map(_.typ)))
      case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
        val initOpArgs2 = initOpArgs.map(rebuildIR(_, env, memo))
        val seqOpArgs2 = seqOpArgs.map(rebuildIR(_, env.promoteScan, memo))
        ApplyScanOp(initOpArgs2, seqOpArgs2,
          aggSig.copy(
            initOpArgs = initOpArgs2.map(_.typ),
            seqOpArgs = seqOpArgs2.map(_.typ)))
      case CollectDistributedArray(contexts, globals, cname, gname, body) =>
        val contexts2 = upcast(rebuildIR(contexts, env, memo), memo.requestedType.lookup(contexts).asInstanceOf[Type])
        val globals2 = upcast(rebuildIR(globals, env, memo), memo.requestedType.lookup(globals).asInstanceOf[Type])
        val body2 = rebuildIR(body, BindingEnv(Env(cname -> contexts2.typ.asInstanceOf[TStream].elementType, gname -> globals2.typ)), memo)
        CollectDistributedArray(contexts2, globals2, cname, gname, body2)
      case _ =>
        ir.copy(ir.children.map {
          case valueIR: IR => rebuildIR(valueIR, env, memo) // FIXME: assert IR does not bind or change env
          case mir: MatrixIR => rebuild(mir, memo)
          case tir: TableIR => rebuild(tir, memo)
          case bmir: BlockMatrixIR => bmir //NOTE Currently no BlockMatrixIRs would have dead fields
        })
    }
  }

  def upcast(ir: IR, rType: Type): IR = {
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
              f.name -> upcast(GetField(ref, f.name), f.typ)
            }
          )
          Let(uid, ir, If(IsNA(ref), NA(ms.typ), ms))
        case ts: TStream =>
          val ra = rType.asInstanceOf[TStream]
          val uid = genUID()
          val ref = Ref(uid, ts.elementType)
          StreamMap(ir, uid, upcast(ref, ra.elementType))
        case ts: TArray =>
          val ra = rType.asInstanceOf[TArray]
          val uid = genUID()
          val ref = Ref(uid, ts.elementType)
          ToArray(StreamMap(ToStream(ir), uid, upcast(ref, ra.elementType)))
        case _: TTuple =>
          val rt = rType.asInstanceOf[TTuple]
          val uid = genUID()
          val ref = Ref(uid, ir.typ)
          val mt = MakeTuple(rt.fields.map { fd =>
            fd.index -> upcast(GetTupleElement(ref, fd.index), fd.typ)
          })
          Let(uid, ir, If(IsNA(ref), NA(mt.typ), mt))
        case _: TDict =>
          val rd = rType.asInstanceOf[TDict]
          ToDict(upcast(ToStream(ir), TArray(rd.elementType)))
        case _: TSet =>
          val rs = rType.asInstanceOf[TSet]
          ToSet(upcast(ToStream(ir), TSet(rs.elementType)))
        case _ => ir
      }

      assert(result.typ == rType, s"${ Pretty(result) }, ${ result.typ }, $rType")
      result
    }
  }

  def upcast(ir: MatrixIR, rType: MatrixType,
    upcastRows: Boolean = true,
    upcastCols: Boolean = true,
    upcastGlobals: Boolean = true,
    upcastEntries: Boolean = true): MatrixIR = {

    if (ir.typ == rType || !(upcastRows || upcastCols || upcastGlobals || upcastEntries))
      ir
    else {
      var mt = ir

      if (ir.typ.rowKey != rType.rowKey) {
        assert(ir.typ.rowKey.startsWith(rType.rowKey))
        mt = MatrixKeyRowsBy(mt, rType.rowKey)
      }

      if (upcastEntries && mt.typ.entryType != rType.entryType)
        mt = MatrixMapEntries(mt, upcast(Ref("g", mt.typ.entryType), rType.entryType))

      if (upcastRows && mt.typ.rowType != rType.rowType)
        mt = MatrixMapRows(mt, upcast(Ref("va", mt.typ.rowType), rType.rowType))

      if (upcastCols && (mt.typ.colType != rType.colType || mt.typ.colKey != rType.colKey)) {
        mt = MatrixMapCols(mt, upcast(Ref("sa", mt.typ.colType), rType.colType),
          if (rType.colKey == mt.typ.colKey) None else Some(rType.colKey))
      }

      if (upcastGlobals && mt.typ.globalType != rType.globalType)
        mt = MatrixMapGlobals(mt, upcast(Ref("global", ir.typ.globalType), rType.globalType))

      mt
    }
  }

  def upcastTable(
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
        table = TableMapRows(table, upcast(Ref("row", table.typ.rowType), rType.rowType))
      }
      if (upcastGlobals && ir.typ.globalType != rType.globalType) {
        table = TableMapGlobals(table,
          upcast(Ref("global", table.typ.globalType), rType.globalType))
      }
      table
    }
  }
}

