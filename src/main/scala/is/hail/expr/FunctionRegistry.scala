package is.hail.expr

import breeze.linalg.DenseVector
import is.hail.annotations.Annotation
import is.hail.asm4s.Code._
import is.hail.asm4s.{Code, _}
import is.hail.expr.CompilationHelp.arrayToWrappedArray
import is.hail.methods._
import is.hail.stats._
import is.hail.utils.EitherIsAMonad._
import is.hail.utils._
import is.hail.variant.{AltAllele, Call, Genotype, Locus, Variant}
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.collection.mutable
import scala.language.higherKinds
import scala.reflect.ClassTag

import org.apache.commons.math3.stat.inference.ChiSquareTest

case class MetaData(docstring: Option[String], args: Seq[(String, String)] = Seq.empty[(String, String)])


object FunctionRegistry {

  sealed trait LookupError {
    def message: String
  }

  sealed case class NotFound(name: String, typ: TypeTag) extends LookupError {
    def message = s"No function found with name `$name' and argument ${ plural(typ.xs.size, "type") } $typ"
  }

  sealed case class Ambiguous(name: String, typ: TypeTag, alternates: Seq[(Int, (TypeTag, Fun))]) extends LookupError {
    def message =
      s"""found ${ alternates.size } ambiguous matches for $typ:
         |  ${ alternates.map(_._2._1).mkString("\n  ") }""".stripMargin
  }

  type Err[T] = Either[LookupError, T]

  private val registry = mutable.HashMap[String, Seq[(TypeTag, Fun, MetaData)]]().withDefaultValue(Seq.empty)

  private val conversions = new mutable.HashMap[(Type, Type), (Int, Transformation[Any, Any])]

  private def lookupConversion(from: Type, to: Type): Option[(Int, Transformation[Any, Any])] = conversions.get(from -> to)

  private val chisq = new ChiSquareTest()

  private def registerConversion[T, U](how: T => U, codeHow: Code[T] => CM[Code[U]], priority: Int = 1)(implicit hrt: HailRep[T], hru: HailRep[U]) {
    val from = hrt.typ
    val to = hru.typ
    require(priority >= 1)
    lookupConversion(from, to) match {
      case Some(_) =>
        throw new RuntimeException(s"The conversion between $from and $to is already bound")
      case None =>
        conversions.put(from -> to, priority -> Transformation[Any, Any](x => how(x.asInstanceOf[T]), x => codeHow(x.asInstanceOf[Code[T]])))
    }
  }

  private def lookup(name: String, typ: TypeTag): Err[Fun] = {

    val matches = registry(name).flatMap { case (tt, f, _) =>
      tt.clear()
      if (tt.xs.size == typ.xs.size) { // FIXME: add check for  to enforce field vs method
        val conversions = (tt.xs, typ.xs).zipped.map { case (l, r) =>
          if (l.isBound) {
            if (l.unify(r))
              Some(None)
            else {
              val conv = lookupConversion(r, l).map(c => Some(c))
              conv
            }
          } else if (l.unify(r)) {
            Some(None)
          } else
            None
        }

        anyFailAllFail[Array, Option[(Int, Transformation[Any, Any])]](conversions)
          .map { arr =>
            if (arr.forall(_.isEmpty))
              0 -> (tt.subst(), f.subst())
            else {
              val arr2 = arr.map(_.getOrElse(0 -> Transformation[Any, Any]((a: Any) => a, (a: Code[Any]) => CM.ret(a))))
              arr2.map(_._1).max -> (tt.subst(), f.subst().convertArgs(arr2.map(_._2)))
            }
          }
      } else
        None
    }.groupBy(_._1).toArray.sortBy(_._1)

    matches.headOption
      .toRight[LookupError](NotFound(name, typ))
      .flatMap { case (priority, it) =>
        assert(it.nonEmpty)
        if (it.size == 1)
          Right(it.head._2._2)
        else {
          assert(priority != 0, s"when it is non-singular, I expect non-zero priority, but priority was $priority and it was $it. name was $name, typ was $typ")
          Left(Ambiguous(name, typ, it))
        }
      }
  }

  private def bind(name: String, typ: TypeTag, f: Fun, md: MetaData) = {
    registry.updateValue(name, Seq.empty, (typ, f, md) +: _)
  }

  def getRegistry() = registry

  def lookupFieldReturnType(typ: Type, typs: Seq[Type], name: String): Err[Type] =
    lookup(name, FieldType(typ +: typs: _*)).map(_.retType)

  def lookupField(typ: Type, typs: Seq[Type], name: String)(lhs: AST, args: Seq[AST]): Err[CM[Code[AnyRef]]] = {
    require(args.isEmpty)

    val m = lookup(name, FieldType(typ +: typs: _*))
    m.map { f =>
      (f match {
        case f: UnaryFun[_, _] =>
          AST.evalComposeCodeM(lhs)(CM.invokePrimitive1(f.asInstanceOf[AnyRef => AnyRef]))
        case f: UnaryFunCode[t, u] =>
          AST.evalComposeCodeM[t](lhs)(f.asInstanceOf[Code[t] => CM[Code[AnyRef]]])
        case fn =>
          throw new RuntimeException(s"Internal hail error, bad binding in function registry for `$name' with argument types $typ, $typs: $fn")
      }).map(Code.checkcast(_)(f.retType.scalaClassTag))
    }
  }

  def call(name: String, args: Seq[AST], argTypes: Seq[Type]): CM[Code[AnyRef]] = {
    import is.hail.expr.CM._

    val m = FunctionRegistry.lookup(name, MethodType(argTypes: _*))
      .valueOr(x => fatal(x.message))

    (m match {
      case aggregator: Arity0Aggregator[_, _] =>
        for (
          aggregationResultThunk <- addAggregation(args(0), aggregator.ctor());
          res <- invokePrimitive0(aggregationResultThunk)
        ) yield res.asInstanceOf[Code[AnyRef]]

      case aggregator: Arity1Aggregator[_, u, _] =>
        for (
          ec <- ec();
          u = args(1).run(ec)();

          _ = (if (u == null)
            fatal(s"Argument evaluated to missing in call to aggregator $name"));

          aggregationResultThunk <- addAggregation(args(0), aggregator.ctor(u.asInstanceOf[u]));
          res <- invokePrimitive0(aggregationResultThunk)
        ) yield res.asInstanceOf[Code[AnyRef]]

      case aggregator: Arity3Aggregator[_, u, v, w, _] =>
        for (
          ec <- ec();
          u = args(1).run(ec)();
          v = args(2).run(ec)();
          w = args(3).run(ec)();

          _ = (if (u == null)
            fatal(s"Argument 1 evaluated to missing in call to aggregator $name"));
          _ = (if (v == null)
            fatal(s"Argument 2 evaluated to missing in call to aggregator $name"));
          _ = (if (w == null)
            fatal(s"Argument 3 evaluated to missing in call to aggregator $name"));


          aggregationResultThunk <- addAggregation(args(0), aggregator.ctor(
            u.asInstanceOf[u],
            v.asInstanceOf[v],
            w.asInstanceOf[w]));
          res <- invokePrimitive0(aggregationResultThunk)
        ) yield res.asInstanceOf[Code[AnyRef]]

      case aggregator: UnaryLambdaAggregator[t, u, v] =>
        val Lambda(_, param, body) = args(1)
        val TFunction(Seq(paramType), _) = argTypes(1)

        for (
          ec <- ec();
          st <- currentSymbolTable();
          (idx, localA) <- ecNewPosition();

          bodyST = args(0).`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => st
          };

          bodyFn = (for (
            fb <- fb();
            bindings = (bodyST.toSeq
              .map { case (name, (i, typ)) =>
                (name, typ, ret(Code.checkcast(fb.arg2.invoke[Int, AnyRef]("apply", i))(typ.scalaClassTag)))
              } :+ ((param, paramType, ret(Code.checkcast(fb.arg2.invoke[Int, AnyRef]("apply", idx))(paramType.scalaClassTag)))));
            res <- bindRepInRaw(bindings)(body.compile())
          ) yield res).runWithDelayedValues(bodyST.toSeq.map { case (name, (_, typ)) => (name, typ) }, ec);

          g = (x: Any) => {
            localA(idx) = x
            bodyFn(localA.asInstanceOf[mutable.ArrayBuffer[AnyRef]])
          };

          aggregationResultThunk <- addAggregation(args(0), aggregator.ctor(g));

          res <- invokePrimitive0(aggregationResultThunk)
        ) yield res.asInstanceOf[Code[AnyRef]]

      case aggregator: BinaryLambdaAggregator[t, u, v, w] =>
        val Lambda(_, param, body) = args(1)
        val TFunction(Seq(paramType), _) = argTypes(1)

        for (
          ec <- ec();
          st <- currentSymbolTable();
          (idx, localA) <- ecNewPosition();

          bodyST = args(0).`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => st
          };

          bodyFn = (for (
            fb <- fb();
            bindings = (bodyST.toSeq
              .map { case (name, (i, typ)) =>
                (name, typ, ret(Code.checkcast(fb.arg2.invoke[Int, AnyRef]("apply", i))(typ.scalaClassTag)))
              } :+ ((param, paramType, ret(Code.checkcast(fb.arg2.invoke[Int, AnyRef]("apply", idx))(paramType.scalaClassTag)))));
            res <- bindRepInRaw(bindings)(body.compile())
          ) yield res).runWithDelayedValues(bodyST.toSeq.map { case (name, (_, typ)) => (name, typ) }, ec);

          g = (x: Any) => {
            localA(idx) = x
            bodyFn(localA.asInstanceOf[mutable.ArrayBuffer[AnyRef]])
          };

          v = args(2).run(ec)();

          _ = (if (v == null)
            fatal(s"Argument evaluated to missing in call to aggregator $name"));

          aggregationResultThunk <- addAggregation(args(0), aggregator.ctor(g, v.asInstanceOf[v]));

          res <- invokePrimitive0(aggregationResultThunk)
        ) yield res.asInstanceOf[Code[AnyRef]]

      case f: UnaryFun[_, _] =>
        AST.evalComposeCodeM(args(0))(invokePrimitive1(f.asInstanceOf[AnyRef => AnyRef]))
      case f: UnarySpecial[_, _] =>
        // FIXME: don't thunk the argument
        args(0).compile().flatMap(invokePrimitive1(x => f.asInstanceOf[(() => AnyRef) => AnyRef](() => x)))
      case f: BinaryFun[_, _, _] =>
        AST.evalComposeCodeM(args(0), args(1))(invokePrimitive2(f.asInstanceOf[(AnyRef, AnyRef) => AnyRef]))
      case f: BinarySpecial[_, _, _] => {
        val g = ((x: AnyRef, y: AnyRef) =>
          f.asInstanceOf[(() => AnyRef, () => AnyRef) => AnyRef](() => x, () => y))

        for (
          t <- args(0).compile();
          u <- args(1).compile();
          result <- invokePrimitive2(g)(t, u))
          yield result
      }
      case f: BinaryLambdaFun[t, _, _] =>
        val Lambda(_, param, body) = args(1)
        val TFunction(Seq(paramType), _) = argTypes(1)
        args(0).`type` match {
          case tagg: TAggregable =>
            if (!tagg.symTab.isEmpty)
              throw new RuntimeException(s"found a non-empty symbol table in a taggregable: $tagg, $tagg.symTab, $name, $args, $argTypes")
          case _ =>
        }

        val g = ((xs: AnyRef, lam: AnyRef) =>
          f(xs.asInstanceOf[t], lam.asInstanceOf[Any => Any]).asInstanceOf[AnyRef])

        for (
          lamc <- createLambda(param, paramType, body.compile());
          res <- AST.evalComposeCodeM(args(0)) { xs =>
            invokePrimitive2[AnyRef, AnyRef, AnyRef](g)(xs, lamc)
          }
        ) yield res
      case f: Arity3LambdaFun[t, _, v, _] =>
        val Lambda(_, param, body) = args(1)
        val TFunction(Seq(paramType), _) = argTypes(1)
        args(0).`type` match {
          case tagg: TAggregable =>
            if (!tagg.symTab.isEmpty)
              throw new RuntimeException(s"found a non-empty symbol table in a taggregable: $tagg, $tagg.symTab, $name, $args, $argTypes")
          case _ =>
        }

        val g = ((xs: AnyRef, lam: AnyRef, y: AnyRef) =>
          f(xs.asInstanceOf[t], lam.asInstanceOf[Any => Any], y.asInstanceOf[v]).asInstanceOf[AnyRef])

        for (
          lamc <- createLambda(param, paramType, body.compile());
          res <- AST.evalComposeCodeM(args(0), args(2)) { (xs, y) =>
            invokePrimitive3[AnyRef, AnyRef, AnyRef, AnyRef](g)(xs, lamc, y)
          }
        ) yield res
      case f: Arity3Fun[_, _, _, _] =>
        AST.evalComposeCodeM(args(0), args(1), args(2))(invokePrimitive3(f.asInstanceOf[(AnyRef, AnyRef, AnyRef) => AnyRef]))
      case f: Arity3Special[_, _, _, _] => {
        val g = ((x: AnyRef, y: AnyRef, z: AnyRef) =>
          f.asInstanceOf[(() => AnyRef, () => AnyRef, () => AnyRef) => AnyRef](() => x, () => y, () => z))

        for (
          t <- args(0).compile();
          u <- args(1).compile();
          v <- args(2).compile();
          result <- invokePrimitive3(g)(t, u, v))
          yield result
      }
      case f: Arity4Fun[_, _, _, _, _] =>
        AST.evalComposeCodeM(args(0), args(1), args(2), args(3))(invokePrimitive4(f.asInstanceOf[(AnyRef, AnyRef, AnyRef, AnyRef) => AnyRef]))
      case f: Arity5Fun[_, _, _, _, _, _] =>
        AST.evalComposeCodeM(args(0), args(1), args(2), args(3), args(4))(invokePrimitive5(f.asInstanceOf[(AnyRef, AnyRef, AnyRef, AnyRef, AnyRef) => AnyRef]))
      case f: UnaryFunCode[t, u] =>
        AST.evalComposeCodeM[t](args(0))(f.asInstanceOf[Code[t] => CM[Code[AnyRef]]])
      case f: BinaryFunCode[t, u, v] =>
        AST.evalComposeCodeM[t, u](args(0), args(1))(f.asInstanceOf[(Code[t], Code[u]) => CM[Code[AnyRef]]])
      case f: BinarySpecialCode[t, u, v] => for (
        a0 <- args(0).compile();
        a1 <- args(1).compile();
        result <- f(a0.asInstanceOf[Code[t]], a1.asInstanceOf[Code[u]]))
        yield result.asInstanceOf[Code[AnyRef]]
      case f: BinaryLambdaAggregatorTransformer[t, _, _] =>
        throw new RuntimeException(s"Internal hail error, aggregator transformation ($name : ${ argTypes.mkString(",") }) in non-aggregator position")
      case f: Arity6Special[_, _, _, _, _, _, _] => {
        val g = ((t: AnyRef, u: AnyRef, v: AnyRef, w: AnyRef, x: AnyRef, y: AnyRef) =>
          f.asInstanceOf[(() => AnyRef, () => AnyRef, () => AnyRef, () => AnyRef, () => AnyRef, () => AnyRef) => AnyRef](() => t, () => u, () => v, () => w, () => x, () => y))

        for (
          t <- args(0).compile();
          u <- args(1).compile();
          v <- args(2).compile();
          w <- args(3).compile();
          x <- args(4).compile();
          y <- args(5).compile();
          result <- invokePrimitive6(g)(t, u, v, w, x, y))
          yield result
      }
      case x =>
        throw new RuntimeException(s"Internal hail error, unexpected Fun type: ${ x.getClass } $x")
    }).map(Code.checkcast(_)(m.retType.scalaClassTag))
  }

  def callAggregatorTransformation(typ: Type, typs: Seq[Type], name: String)(lhs: AST, args: Seq[AST]): CMCodeCPS[AnyRef] = {
    import is.hail.expr.CM._

    require(typs.length == args.length)

    val m = FunctionRegistry.lookup(name, MethodType(typ +: typs: _*))
      .valueOr(x => fatal(x.message))

    m match {
      case f: BinaryLambdaAggregatorTransformer[t, _, _] =>
        val Lambda(_, param, body) = args(0)
        val TFunction(Seq(paramType), _) = typs(0)

      { (k: Code[AnyRef] => CM[Code[Unit]]) =>
        for (
          st <- currentSymbolTable();

          bodyST = lhs.`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => st
          };

          fb <- fb();

          externalBindings = bodyST.toSeq.map { case (name, (i, typ)) =>
            (name, typ, ret(Code.checkcast(fb.arg2.invoke[Int, AnyRef]("apply", i))(typ.scalaClassTag)))
          };

          g = { (_x: Code[AnyRef]) =>
            for (
              (stx, x) <- memoize(_x);
              bindings = externalBindings :+ ((param, paramType, ret(x)));
              cbody <- bindRepInRaw(bindings)(body.compile())
            ) yield Code(stx, cbody)
          }: (Code[AnyRef] => CM[Code[AnyRef]]);

          res <- lhs.compileAggregator() { (t: Code[AnyRef]) =>
            f.fcode(t, g)(k)
          }
        ) yield res
      }
      case _ =>
        throw new RuntimeException(s"Internal hail error, non-aggregator transformation, `$name' with argument types $typ, $typs, found in aggregator position")
    }
  }

  def lookupMethodReturnType(typ: Type, typs: Seq[Type], name: String): Err[Type] =
    lookup(name, MethodType(typ +: typs: _*)).map(_.retType)

  def lookupFunReturnType(name: String, typs: Seq[Type]): Err[Type] =
    lookup(name, FunType(typs: _*)).map(_.retType)

  def registerField[T, U](name: String, impl: T => U, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FieldType(hrt.typ), UnaryFun[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerFieldCode[T, U](name: String, impl: (Code[T]) => CM[Code[U]], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FieldType(hrt.typ), UnaryFunCode[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerMethod[T, U](name: String, impl: T => U, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryFun[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerMethodCode[T, U](name: String, impl: (Code[T]) => CM[Code[U]], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryFunCode[T, U](hru.typ, (ct) => impl(ct)), MetaData(Option(docstring), argNames))
  }

  def registerMethodSpecial[T, U](name: String, impl: (() => Any) => U, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnarySpecial[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerMethod[T, U, V](name: String, impl: (T, U) => V, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerMethodCode[T, U, V](name: String, impl: (Code[T], Code[U]) => CM[Code[V]], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinaryFunCode[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerMethodSpecial[T, U, V](name: String, impl: (() => Any, () => Any) => V, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinarySpecial[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerLambdaMethod[T, U, V](name: String, impl: (T, (Any) => Any) => V, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    val m = BinaryLambdaFun[T, U, V](hrv.typ, impl)
    bind(name, MethodType(hrt.typ, hru.typ), m, MetaData(Option(docstring), argNames))
  }

  def registerLambdaMethod[T, U, V, W](name: String, impl: (T, (Any) => Any, V) => W, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    val m = Arity3LambdaFun[T, U, V, W](hrw.typ, impl)
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), m, MetaData(Option(docstring), argNames))
  }

  def registerLambdaAggregatorTransformer[T, U, V](name: String, impl: (CPS[Any], (Any) => Any) => CPS[V],
      codeImpl: (Code[AnyRef], Code[AnyRef] => CM[Code[AnyRef]]) => CMCodeCPS[AnyRef],
      docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    val m = BinaryLambdaAggregatorTransformer[T, U, V](hrv.typ, impl, codeImpl)
    bind(name, MethodType(hrt.typ, hru.typ), m, MetaData(Option(docstring), argNames))
  }

  def registerMethod[T, U, V, W](name: String, impl: (T, U, V) => W, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ, impl), MetaData(Option(docstring), argNames))
  }

  def register[T, U](name: String, impl: T => U, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryFun[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerCode[T, U](name: String, impl: Code[T] => CM[Code[U]], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryFunCode[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerSpecial[T, U](name: String, impl: (() => Any) => U, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnarySpecial[T, U](hru.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerUnaryNAFilteredCollectionMethod[T, U](name: String, impl: TraversableOnce[T] => U, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(TArray(hrt.typ)), UnaryFun[IndexedSeq[_], U](hru.typ, { (ts: IndexedSeq[_]) =>
      impl(ts.filter(t => t != null).map(_.asInstanceOf[T]))
    }), MetaData(Option(docstring), argNames))
    bind(name, MethodType(TSet(hrt.typ)), UnaryFun[Set[_], U](hru.typ, { (ts: Set[_]) =>
      impl(ts.filter(t => t != null).map(_.asInstanceOf[T]))
    }), MetaData(Option(docstring), argNames))
  }

  def register[T, U, V](name: String, impl: (T, U) => V, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerCode[T, U, V](name: String, impl: (Code[T], Code[U]) => CM[Code[V]], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryFunCode[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerSpecial[T, U, V](name: String, impl: (() => Any, () => Any) => V, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinarySpecial[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerSpecialCode[T, U, V](name: String, impl: (Code[T], Code[U]) => CM[Code[V]], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinarySpecialCode[T, U, V](hrv.typ, impl), MetaData(Option(docstring), argNames))
  }

  def register[T, U, V, W](name: String, impl: (T, U, V) => W, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerSpecial[T, U, V, W](name: String, impl: (() => Any, () => Any, () => Any) => W, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3Special[T, U, V, W](hrw.typ, impl), MetaData(Option(docstring), argNames))
  }

  def register[T, U, V, W, X](name: String, impl: (T, U, V, W) => X, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity4Fun[T, U, V, W, X](hrx.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerSpecial[T, U, V, W, X, Y, Z](name: String, impl: (() => Any, () => Any, () => Any, () => Any, () => Any, () => Any) => Z, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X], hry: HailRep[Y], hrz: HailRep[Z]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ, hrx.typ, hry.typ), Arity6Special[T, U, V, W, X, Y, Z](hrz.typ, impl), MetaData(Option(docstring), argNames))
  }

  def register[T, U, V, W, X, Y](name: String, impl: (T, U, V, W, X) => Y, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X], hry: HailRep[Y]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ, hrx.typ), Arity5Fun[T, U, V, W, X, Y](hry.typ, impl), MetaData(Option(docstring), argNames))
  }

  def registerAnn[T](name: String, t: TStruct, impl: T => Annotation, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T]) = {
    register(name, impl, docstring, argNames: _*)(hrt, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U](name: String, t: TStruct, impl: (T, U) => Annotation, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    register(name, impl, docstring, argNames: _*)(hrt, hru, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V](name: String, t: TStruct, impl: (T, U, V) => Annotation, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    register(name, impl, docstring, argNames: _*)(hrt, hru, hrv, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V, W](name: String, t: TStruct, impl: (T, U, V, W) => Annotation, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    register(name, impl, docstring, argNames: _*)(hrt, hru, hrv, hrw, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V, W, X](name: String, t: TStruct, impl: (T, U, V, W, X) => Annotation, docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    register(name, impl, docstring, argNames: _*)(hrt, hru, hrv, hrw, hrx, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAggregator[T, U](name: String, ctor: () => TypedAggregator[U], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), Arity0Aggregator[T, U](hru.typ, ctor), MetaData(Option(docstring), argNames))
  }

  def registerLambdaAggregator[T, U, V](name: String, ctor: ((Any) => Any) => TypedAggregator[V], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), UnaryLambdaAggregator[T, U, V](hrv.typ, ctor), MetaData(Option(docstring), argNames))
  }

  def registerLambdaAggregator[T, U, V, W](name: String, ctor: ((Any) => Any, V) => TypedAggregator[W], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), BinaryLambdaAggregator[T, U, V, W](hrw.typ, ctor), MetaData(Option(docstring), argNames))
  }

  def registerAggregator[T, U, V](name: String, ctor: (U) => TypedAggregator[V], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), Arity1Aggregator[T, U, V](hrv.typ, ctor), MetaData(Option(docstring), argNames))
  }

  def registerAggregator[T, U, V, W, X](name: String, ctor: (U, V, W) => TypedAggregator[X], docstring: String, argNames: (String, String)*)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity3Aggregator[T, U, V, W, X](hrx.typ, ctor), MetaData(Option(docstring), argNames))
  }

  val TT = TVariable("T")
  val TU = TVariable("U")
  val TV = TVariable("V")

  val TTBoxed = TVariable("T")

  val TTHr = new HailRep[Any] {
    def typ = TT
  }
  val TUHr = new HailRep[Any] {
    def typ = TU
  }
  val TVHr = new HailRep[Any] {
    def typ = TV
  }
  val BoxedTTHr = new HailRep[AnyRef] {
    def typ = TTBoxed
  }

  private def nonceToNullable[T : TypeInfo, U >: Null](check: Code[T] => Code[Boolean], v: Code[T], ifPresent: Code[T] => Code[U]): CM[Code[U]] = for (
    (stx, x) <- CM.memoize(v)
  ) yield Code(stx, check(x).mux(Code._null[U], ifPresent(x)))

  import is.hail.variant.Call._

  registerField("gt", { (c: Call) => c }, "the integer ``gt = k*(k+1)/2 + j`` for call ``j/k`` (0 = 0/0, 1 = 0/1, 2 = 1/1, 3 = 0/2, etc.).")(callHr, boxedInt32Hr)
  registerMethodSpecial("gtj", { (c: () => Any) => gtj(c().asInstanceOf[Call]) }, "the index of allele ``j`` for call ``j/k`` (0 = ref, 1 = first alt allele, etc.).")(callHr, boxedInt32Hr)
  registerMethodSpecial("gtk", { (c: () => Any) => gtk(c().asInstanceOf[Call]) }, "the index of allele ``k`` for call ``j/k`` (0 = ref, 1 = first alt allele, etc.).")(callHr, boxedInt32Hr)
  registerMethodSpecial("isHomRef", { (c: () => Any) => isHomRef(c().asInstanceOf[Call]) }, "True if this call is ``0/0``.")(callHr, boolHr)
  registerMethodSpecial("isHet", { (c: () => Any) => isHet(c().asInstanceOf[Call]) }, "True if this call is heterozygous.")(callHr, boolHr)
  registerMethodSpecial("isHomVar", { (c: () => Any) => isHomVar(c().asInstanceOf[Call]) }, "True if this call is ``j/j`` with ``j>0``.")(callHr, boolHr)
  registerMethodSpecial("isCalledNonRef", { (c: () => Any) => isCalledNonRef(c().asInstanceOf[Call]) }, "True if either ``isHet`` or ``isHomVar`` is true.")(callHr, boolHr)
  registerMethodSpecial("isHetNonRef", { (c: () => Any) => isHetNonRef(c().asInstanceOf[Call]) }, "True if this call is ``j/k`` with ``j>0``.")(callHr, boolHr)
  registerMethodSpecial("isHetRef", { (c: () => Any) => isHetRef(c().asInstanceOf[Call]) }, "True if this call is ``0/k`` with ``k>0``.")(callHr, boolHr)
  registerMethodSpecial("isCalled", { (c: () => Any) => isCalled(c().asInstanceOf[Call]) }, "True if the call is not ``./.``.")(callHr, boolHr)
  registerMethodSpecial("isNotCalled", { (c: () => Any) => isNotCalled(c().asInstanceOf[Call]) }, "True if the call is ``./.``.")(callHr, boolHr)
  registerMethodSpecial("nNonRefAlleles", { (c: () => Any) => nNonRefAlleles(c().asInstanceOf[Call]) }, "the number of called alternate alleles.")(callHr, boxedInt32Hr)
  registerMethodSpecial("toGenotype", { (c: () => Any) => toGenotype(c().asInstanceOf[Call]) }, "Convert this call to a Genotype.")(callHr, genotypeHr)
  registerMethodSpecial("oneHotAlleles", { (c: () => Any, v: () => Any) =>
    val call = c().asInstanceOf[Call]
    val variant = v().asInstanceOf[Variant]
    if (call != null && variant != null)
      oneHotAlleles(call, variant)
    else
      null
  },
    """
    Produce an array of called counts for each allele in the variant (including reference). For example, calling this function with a biallelic variant on hom-ref, het, and hom-var calls will produce ``[2, 0]``, ``[1, 1]``, and ``[0, 2]`` respectively.
    """,
    "v" -> ":ref:`variant`")(callHr, variantHr, arrayHr(int32Hr))
  registerMethodSpecial("oneHotGenotype", { (c: () => Any, v: () => Any) =>
    val call = c().asInstanceOf[Call]
    val variant = v().asInstanceOf[Variant]
    if (call != null && variant != null)
      oneHotGenotype(call, variant)
    else
      null
  },
    """
    Produces an array with one element for each possible genotype in the variant, where the called genotype is 1 and all else 0. For example, calling this function with a biallelic variant on hom-ref, het, and hom-var calls will produce ``[1, 0, 0]``, ``[0, 1, 0]``, and ``[0, 0, 1]`` respectively.
    """,
    "v" -> ":ref:`variant`")(callHr, variantHr, arrayHr(int32Hr))

  registerFieldCode("gt", { (x: Code[Genotype]) =>
    nonceToNullable[Int, java.lang.Integer](_.ceq(-1), x.invoke[Int]("_unboxedGT"), boxInt(_))
  }, "the integer ``gt = k*(k+1)/2 + j`` for call ``j/k`` (0 = 0/0, 1 = 0/1, 2 = 1/1, 3 = 0/2, etc.).")
  registerMethod("call", { (x: Genotype) => Genotype.call(x) }, "the integer ``gt = k*(k+1)/2 + j`` for call ``j/k`` (0 = 0/0, 1 = 0/1, 2 = 1/1, 3 = 0/2, etc.).")(genotypeHr, callHr)
  registerMethod("gtj", { (x: Genotype) =>
    val gt = x._unboxedGT
    if (gt == -1)
      null
    else
      box(Genotype.gtPair(gt).j)
  }, "the index of allele ``j`` for call ``j/k`` (0 = ref, 1 = first alt allele, etc.).")
  registerMethod("gtk", { (x: Genotype) =>
    val gt = x._unboxedGT
    if (gt == -1)
      null
    else
      box(Genotype.gtPair(gt).k)
  }, "the index of allele ``k`` for call ``j/k`` (0 = ref, 1 = first alt allele, etc.).")
  registerFieldCode("ad", { (x: Code[Genotype]) =>
    CM.ret(arrayToWrappedArray(x.invoke[Array[Int]]("_unboxedAD")))
  }, "allelic depth for each allele.")
  registerFieldCode("dp", { (x: Code[Genotype]) =>
    nonceToNullable[Int, java.lang.Integer](_.ceq(-1), x.invoke[Int]("_unboxedDP"), boxInt)
  }, "the total number of informative reads.")
  registerMethodCode("od", { (x: Code[Genotype]) =>
    CM.ret(x.invoke[Boolean]("hasOD").mux(boxInt(x.invoke[Int]("od_")),Code._null[java.lang.Integer]))
  }, "``od = dp - ad.sum``.")
  registerFieldCode("gq", { (x: Code[Genotype]) =>
    nonceToNullable[Int, java.lang.Integer](_.ceq(-1), x.invoke[Int]("_unboxedGQ"), boxInt)
  }, "the difference between the two smallest PL entries.")
  registerFieldCode("pl", { (x: Code[Genotype]) =>
    CM.ret(arrayToWrappedArray(Code.invokeStatic[Genotype, Genotype, Array[Int]]("unboxedPL", x)))
  }, """
     phred-scaled normalized genotype likelihood values. The conversion between
     ``g.pl`` (Phred-scaled likelihoods) and ``g.gp`` (linear-scaled
     probabilities) assumes a uniform prior.
     """)
  registerFieldCode("gp", { (x: Code[Genotype]) =>
    CM.ret(arrayToWrappedArray(Code.invokeStatic[Genotype, Genotype, Array[Double]]("unboxedGP", x)))
  }, "the linear-scaled probabilities.")
  registerFieldCode("dosage", { (x: Code[Genotype]) =>
    nonceToNullable[Double, java.lang.Double](_.ceq(-1d), Code.invokeStatic[Genotype, Genotype, Double]("unboxedDosage", x), boxDouble)
  }, "the expected number of non-reference alleles based on genotype probabilities.")
  registerMethodCode("isHomRef", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isHomRef", x)))
  }, "True if this call is ``0/0``.")
  registerMethodCode("isHet", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isHet", x)))
  }, "True if this call is heterozygous.")
  registerMethodCode("isHomVar", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isHomVar", x)))
  }, "True if this call is ``j/j`` with ``j>0``.")
  registerMethodCode("isCalledNonRef", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isCalledNonRef", x)))
  }, "True if either ``g.isHet`` or ``g.isHomVar`` is true.")
  registerMethodCode("isHetNonRef", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isHetNonRef", x)))
  }, "True if this call is ``j/k`` with ``j>0``.")
  registerMethodCode("isHetRef", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isHetRef", x)))
  }, "True if this call is ``0/k`` with ``k>0``.")
  registerMethodCode("isCalled", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isCalled", x)))
  }, "True if the genotype is not ``./.``.")
  registerMethodCode("isNotCalled", { (x: Code[Genotype]) =>
    CM.ret(boxBoolean(Code.invokeStatic[Genotype, Genotype, Boolean]("isNotCalled", x)))
  }, "True if the genotype is ``./.``.")
  registerMethodCode("nNonRefAlleles", { (x: Code[Genotype]) => for (
    (stg, g) <- CM.memoize(x)
  ) yield Code(stg,
    Code.invokeStatic[Genotype, Genotype, Boolean]("hasNNonRefAlleles", g)
      .mux(boxInt(Code.invokeStatic[Genotype, Genotype, Int]("nNonRefAlleles_", g)), Code._null))
  }, "the number of called alternate alleles.")
  registerMethodCode("pAB", { (x: Code[Genotype]) => for (
    (stg, g) <- CM.memoize(x)
  ) yield Code(stg,
      Code.invokeStatic[Genotype, Genotype, Boolean]("hasPAB", g)
      .mux(boxDouble(Code.invokeStatic[Genotype, Genotype, Double, Double]("pAB_", g, 0.5)), Code._null))
  }, "p-value for pulling the given allelic depth from a binomial distribution with mean 0.5.  Missing if the call is not heterozygous.")

  private def intArraySumCode(a: Code[Array[Int]]): CM[Code[Int]] = for (
    (starr, arr) <- CM.memoize(a);
    i <- CM.newLocal[Int];
    s <- CM.newLocal[Int]
  ) yield Code(
    starr,
    i.store(0),
    s.store(0),
    Code.whileLoop(i < arr.length(), Code(s.store(s + arr(i)), i.store(i + 1))),
    s
  )

  registerMethodCode("fractionReadsRef", { (x: Code[Genotype]) => for (
    (stad, ad) <- CM.memoize(Code.invokeStatic[Genotype, Genotype, Array[Int]]("unboxedAD", x));
    (stsum, sum) <- CM.memoize(intArraySumCode(ad))
  ) yield Code(stad, stsum, sum.ceq(0).mux(Code._null, boxDouble(ad(0).toD / sum.toD)))
  }, "the ratio of ref reads to the sum of all *informative* reads.")
  registerFieldCode("fakeRef",
    { (x: Code[Genotype]) => CM.ret(boxBoolean(x.invoke[Boolean]("fakeRef"))) },
    "True if this genotype was downcoded in :py:meth:`~hail.VariantDataset.split_multi`.  This can happen if a ``1/2`` call is split to ``0/1``, ``0/1``.")
  registerFieldCode("isLinearScale",
    { (x: Code[Genotype]) => CM.ret(boxBoolean(x.invoke[Boolean]("isLinearScale"))) },
    "True if the data was imported from :py:meth:`~hail.HailContext.import_gen` or :py:meth:`~hail.HailContext.import_bgen`.")
  registerFieldCode("contig",
    { (x: Code[Variant]) => CM.ret(x.invoke[String]("contig")) },
    "String representation of contig, exactly as imported. *NB: Hail stores contigs as strings. Use double-quotes when checking contig equality.*")
  registerFieldCode("start",
    { (x: Code[Variant]) => CM.ret(boxInt(x.invoke[Int]("start"))) },
    "SNP position or start of an indel.")
  registerFieldCode("ref",
    { (x: Code[Variant]) => CM.ret(x.invoke[String]("ref")) },
    "Reference allele sequence.")
  registerFieldCode("altAlleles",
    { (x: Code[Variant]) => CM.ret(x.invoke[IndexedSeq[AltAllele]]("altAlleles")) },
    "The :ref:`alternate alleles <altallele>`.")
  registerMethod("nAltAlleles", { (x: Variant) => x.nAltAlleles }, "Number of alternate alleles, equal to ``nAlleles - 1``.")
  registerMethod("nAlleles", { (x: Variant) => x.nAlleles }, "Number of alleles.")
  registerMethod("isBiallelic", { (x: Variant) => x.isBiallelic }, "True if `v` has one alternate allele.")
  registerMethod("nGenotypes", { (x: Variant) => x.nGenotypes }, "Number of genotypes.")
  registerMethod("inXPar", { (x: Variant) => x.inXPar }, "True if chromosome is X and start is in pseudoautosomal region of X.")
  registerMethod("inYPar", { (x: Variant) => x.inYPar }, "True if chromosome is Y and start is in pseudoautosomal region of Y. *NB: most callers assign variants in PAR to X.*")
  registerMethod("inXNonPar", { (x: Variant) => x.inXNonPar }, "True if chromosome is X and start is not in pseudoautosomal region of X.")
  registerMethod("inYNonPar", { (x: Variant) => x.inYNonPar }, "True if chromosome is Y and start is not in pseudoautosomal region of Y.")
  // assumes biallelic
  registerMethod("alt", { (x: Variant) => x.alt }, "Alternate allele sequence.  **Assumes biallelic.**")
  registerMethod("altAllele", { (x: Variant) => x.altAllele }, "The :ref:`alternate allele <altallele>`.  **Assumes biallelic.**")
  registerMethod("locus", { (x: Variant) => x.locus }, "Chromosomal locus (chr, pos) of this variant")
  registerMethod("isAutosomal", { (x: Variant) => x.isAutosomal }, "True if chromosome is not X, not Y, and not MT.")
  registerField("contig", { (x: Locus) => x.contig }, "String representation of contig.")
  registerField("position", { (x: Locus) => x.position }, "Chromosomal position.")
  registerField("start", { (x: Interval[Locus]) => x.start }, ":ref:`locus` at the start of the interval (inclusive).")
  registerField("end", { (x: Interval[Locus]) => x.end }, ":ref:`locus` at the end of the interval (exclusive).")
  registerField("ref", { (x: AltAllele) => x.ref }, "Reference allele base sequence.")
  registerField("alt", { (x: AltAllele) => x.alt }, "Alternate allele base sequence.")
  registerMethod("isSNP", { (x: AltAllele) => x.isSNP }, "True if ``v.ref`` and ``v.alt`` are the same length and differ in one position.")
  registerMethod("isMNP", { (x: AltAllele) => x.isMNP }, "True if ``v.ref`` and ``v.alt`` are the same length and differ in more than one position.")
  registerMethod("isIndel", { (x: AltAllele) => x.isIndel }, "True if an insertion or a deletion.")
  registerMethod("isInsertion", { (x: AltAllele) => x.isInsertion }, "True if ``v.alt`` begins with and is longer than ``v.ref``.")
  registerMethod("isDeletion", { (x: AltAllele) => x.isDeletion }, "True if ``v.ref`` begins with and is longer than ``v.alt``.")
  registerMethod("isStar", { (x: AltAllele) => x.isStar }, "True if ``v.alt`` is ``*``.")
  registerMethod("isComplex", { (x: AltAllele) => x.isComplex }, "True if not a SNP, MNP, star, insertion, or deletion.")
  registerMethod("isTransition", { (x: AltAllele) => x.isTransition }, "True if a purine-purine or pyrimidine-pyrimidine SNP.")
  registerMethod("isTransversion", { (x: AltAllele) => x.isTransversion }, "True if a purine-pyrimidine SNP.")
  registerMethod("category", { (x: AltAllele) => x.altAlleleType.toString }, "the alt allele type, i.e one of SNP, Insertion, Deletion, Star, MNP, Complex")

  registerMethod("length", { (x: String) => x.length }, "Length of the string.")

  val sumDocstring = "Sum of all elements in the collection."
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Int]) => x.sum }, sumDocstring)
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Long]) => x.sum }, sumDocstring)
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Float]) => x.sum }, sumDocstring)
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Double]) => x.sum }, sumDocstring)

  val productDocstring = "Product of all elements in the collection (returns 1 if empty)."
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Int]) => x.product }, productDocstring)
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Long]) => x.product }, productDocstring)
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Float]) => x.product }, productDocstring)
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Double]) => x.product }, productDocstring)

  val minDocstring = "Smallest element in the collection."
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(x.min) else null }, minDocstring)(int32Hr, boxedInt32Hr)
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(x.min) else null }, minDocstring)(int64Hr, boxedInt64Hr)
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(x.min) else null }, minDocstring)(float32Hr, boxedFloat32Hr)
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(x.min) else null }, minDocstring)(float64Hr, boxedFloat64Hr)

  val maxDocstring = "Largest element in the collection."
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(x.max) else null }, maxDocstring)(int32Hr, boxedInt32Hr)
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(x.max) else null }, maxDocstring)(int64Hr, boxedInt64Hr)
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(x.max) else null }, maxDocstring)(float32Hr, boxedFloat32Hr)
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(x.max) else null }, maxDocstring)(float64Hr, boxedFloat64Hr)

  val medianDocstring = "Median value of the collection."
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null }, medianDocstring)(int32Hr, boxedInt32Hr)
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null }, medianDocstring)(int64Hr, boxedInt64Hr)
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null }, medianDocstring)(float32Hr, boxedFloat32Hr)
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null }, medianDocstring)(float64Hr, boxedFloat64Hr)

  val meanDocstring = "Mean value of the collection."
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null }, meanDocstring)(int32Hr, boxedFloat64Hr)
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null }, meanDocstring)(int64Hr, boxedFloat64Hr)
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null }, meanDocstring)(float32Hr, boxedFloat64Hr)
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null }, meanDocstring)(float64Hr, boxedFloat64Hr)

  register("range", { (x: Int) => 0 until x: IndexedSeq[Int] },
    """
    Generate an ``Array`` with values in the interval ``[0, stop)``.

    .. code-block:: text
        :emphasize-lines: 2

        let r = range(3) in r
        result: [0, 1, 2]
    """,
    "stop" -> "Number of integers (whole numbers) to generate, starting from zero.")
  register("range", { (x: Int, y: Int) => x until y: IndexedSeq[Int] },
    """
    Generate an ``Array`` with values in the interval ``[start, stop)``.

    .. code-block:: text
        :emphasize-lines: 2

        let r = range(5, 8) in r
        result: [5, 6, 7]
    """,
    "start" -> "Starting number of the sequence.",
    "stop" -> "Generate numbers up to, but not including this number.")
  register("range", { (x: Int, y: Int, step: Int) => x until y by step: IndexedSeq[Int] },
    """
    Generate an ``Array`` with values in the interval ``[start, stop)`` in increments of step.

    .. code-block:: text
        :emphasize-lines: 2

        let r = range(0, 5, 2) in r
        result: [0, 2, 4]
    """,
    "start" -> "Starting number of the sequence.",
    "stop" -> "Generate numbers up to, but not including this number.",
    "step" -> "Difference between each number in the sequence.")

  register("Call", { (gt: java.lang.Integer) => gt },
    """
    Construct a :ref:`call` from an integer.
    """, "gt" -> "integer")(boxedInt32Hr, callHr)

  registerSpecial("Genotype", { (vF: () => Any, dosF: () => Any) =>
    val v = vF()
    val dos = dosF()

    if (v == null)
      throw new HailException("The first argument to Genotype, the Variant, must not be NA.")

    Genotype(v.asInstanceOf[Variant].nAlleles, if (dos == null) null else dos.asInstanceOf[IndexedSeq[Double]].toArray)
  },
    """
    Construct a :ref:`genotype` object from a variant and an array of genotype probabilities.

    .. code-block:: text
        :emphasize-lines: 3

        let v = Variant("7:76324539:A:G") and prob = [0.2, 0.7, 0.1] and
          g = Genotype(v, prob) in g.isHet()
        result: true
    """, "v" -> "Variant", "prob" -> "Genotype probabilities")(variantHr, arrayHr[Double], genotypeHr)

  registerSpecial("Genotype", { (vF: () => Any, gtF: () => Any, dosF: () => Any) =>
    val v = vF()
    val dos = dosF()

    if (v == null)
      throw new HailException("The first argument to Genotype, the Variant, must not be NA.")

    Genotype(v.asInstanceOf[Variant].nAlleles, gtF().asInstanceOf[Call], if (dos == null) null else dos.asInstanceOf[IndexedSeq[Double]].toArray)
  },
    """
    Construct a :ref:`genotype` object from a variant, a genotype call, and an array of genotype probabilities.

    .. code-block:: text
        :emphasize-lines: 3

        let v = Variant("7:76324539:A:G") and gt = Call(0) and prob = [0.8, 0.1, 0.1] and
          g = Genotype(v, gt, prob) in g.isHomRef()
        result: true
    """, "v" -> "Variant", "gt" -> "Genotype call integer", "prob" -> "Genotype probabilities")(variantHr, callHr, arrayHr[Double], genotypeHr)

  registerSpecial("Genotype", { (gtF: () => Any) =>
    Genotype(gtF().asInstanceOf[java.lang.Integer])
  },
    """
    Construct a :ref:`genotype` object from a genotype call.

    .. code-block:: text
        :emphasize-lines: 3

        Genotype(Call(0)).isHomRef()
        result: true
    """, "gt" -> "Genotype call integer")(callHr, genotypeHr)

  registerSpecial("Genotype", { (vF: () => Any, gtF: () => Any, adF: () => Any, dpF: () => Any, gqF: () => Any, plF: () => Any) =>
    val v = vF()
    val ad = adF()
    val pl = plF()

    if (v == null)
      throw new HailException("The first argument to Genotype, the Variant, must not be NA.")

    Genotype(v.asInstanceOf[Variant].nAlleles, gtF().asInstanceOf[Call],
      if (ad == null) null else ad.asInstanceOf[IndexedSeq[Int]].toArray, dpF().asInstanceOf[java.lang.Integer],
      gqF().asInstanceOf[java.lang.Integer], if (pl == null) null else pl.asInstanceOf[IndexedSeq[Int]].toArray)
  },
    """
    Construct a :ref:`genotype` object by specifying the variant, call, allelic depths, depth, genotype quality, and phred-scaled likelihoods.

    .. code-block:: text
        :emphasize-lines: 4

        let v = Variant("7:76324539:A:G") and call = Call(0) and
          ad = [10, 0] and dp = 10 and gq = 20 and pl = [0, 10, 100] and
          g = Genotype(v, call, ad, dp, gq, pl) in g.isHomRef()
        result: true
    """,
    "v" -> "Variant", "c" -> "Call", "ad" -> "Allelic depths", "dp" -> "Depth", "gq" -> "Genotype quality", "pl" -> "Phred-scaled likelihoods"
  )(variantHr, callHr, arrayHr[Int], boxedInt32Hr, boxedInt32Hr, arrayHr[Int], genotypeHr)

  register("Variant", { (x: String) => Variant.parse(x) },
    """
    Construct a :ref:`variant` object.

    .. code-block:: text
        :emphasize-lines: 2

        let v = Variant("7:76324539:A:G") in v.contig
        result: "7"
    """,
    "s" -> "String of the form ``CHR:POS:REF:ALT`` or ``CHR:POS:REF:ALT1,ALT2...ALTN`` specifying the contig, position, reference and alternate alleles.")
  register("Variant", { (x: String, y: Int, z: String, a: String) => Variant(x, y, z, a) },
    """
    Construct a :ref:`variant` object.

    .. code-block:: text
        :emphasize-lines: 2

        let v = Variant("2", 13427, "T", "G") in v.ref
        result: "T"
    """,
    "contig" -> "String representation of contig.",
    "pos" -> "SNP position or start of an indel.",
    "ref" -> "Reference allele sequence.",
    "alt" -> "Alternate allele sequence.")
  register("Variant", { (x: String, y: Int, z: String, a: IndexedSeq[String]) => Variant(x, y, z, a.toArray) },
    """
    Construct a :ref:`variant` object.

    .. code-block:: text
        :emphasize-lines: 2

        let v = Variant("1", 25782743, "A", Array("T", "TA")) in v.ref
        result: "A"
    """,
    "contig" -> "String representation of contig.",
    "pos" -> "SNP position or start of an indel.",
    "ref" -> "Reference allele sequence.",
    "alts" -> "Array of alternate allele sequences."
  )

  register("Dict", { (keys: IndexedSeq[Annotation], values: IndexedSeq[Annotation]) =>
    if (keys.length != values.length)
      fatal(s"mismatch between length of keys (${ keys.length }) and values (${ values.length })")
    keys.zip(values).toMap
  }, "Construct a Dict from an array of keys and an array of values.",
    "keys" -> "Keys of Dict.",
    "values" -> "Values of Dict.")(arrayHr(TTHr), arrayHr(TUHr), dictHr(TTHr, TUHr))

  val combineVariantsStruct = TStruct(Array(("variant", TVariant, "Resulting combined variant."),
    ("laIndices", TDict(TInt32, TInt32), "Mapping from new to old allele index for the left variant."),
    ("raIndices", TDict(TInt32, TInt32), "Mapping from new to old allele index for the right variant.")
  ).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })

  registerAnn("combineVariants",
    combineVariantsStruct, { (left: Variant, right: Variant) =>
      if (left.contig != right.contig || left.start != right.start)
        fatal(s"Only variants with the same contig and position can be combined. Left was $left, right was $right.")

      val (longer, shorter, swapped) = if (left.ref.length > right.ref.length) (left, right, false) else (right, left, true)
      val ref_diff = longer.ref.substring(shorter.ref.length)

      if (longer.ref.substring(0, shorter.ref.length) != shorter.ref)
        fatal(s"Variants ref bases mismatch in combineVariants. Left ref: ${ left.ref }, right ref: ${ right.ref }")

      val long_alleles_index = longer.altAlleles.map(_.alt).zipWithIndex.toMap
      val short_alleles_index = mutable.Map[Int, Int](0 -> 0)
      val short_alleles = new mutable.ArrayBuffer[AltAllele](initialSize = shorter.nAltAlleles)

      (0 until shorter.nAltAlleles).foreach({
        i =>
          val alt = shorter.altAlleles(i).alt + ref_diff
          long_alleles_index.get(alt) match {
            case Some(ai) => short_alleles_index(ai + 1) = i + 1
            case None => short_alleles += AltAllele(longer.ref, alt)
              short_alleles_index(longer.nAltAlleles + short_alleles.length) = i + 1
          }
      })

      val newVariant = longer.copy(altAlleles = longer.altAlleles ++ short_alleles)
      if (swapped)
        Annotation(newVariant, short_alleles_index.toMap, (0 to longer.nAltAlleles).zipWithIndex.toMap)
      else
        Annotation(newVariant, (0 to longer.nAltAlleles).zipWithIndex.toMap, short_alleles_index.toMap)
    },
    """
    Combines the alleles of two variants at the same locus, making sure that ref and alt alleles are represented uniformely.
    In addition to the resulting variant containing all alleles, this function also returns the mapping from the old to the new allele indices.
    Note that this mapping counts the reference allele always contains the reference allele mapping 0 -> 0.

          .. code-block:: text
              :emphasize-lines: 2

              let left = Variant("1:1000:AT:A,CT") and right = Variant("1:1000:A:C,AGG") in combineVariants(left,right)
              result: Struct{'variant': Variant(contig=1, start=1000, ref=AT, alts=[AltAllele(ref=AT, alt=A), AltAllele(ref=AT, alt=CT), AltAllele(ref=AT, alt=AGGT)]), 'laIndices': {0: 0, 1: 1, 2: 2}, 'raIndices': {0:0, 2: 1, 3: 2}}

    """,
    "left" -> "Left variant to combine.",
    "right" -> "Right variant to combine.")

  register("Locus", { (x: String) =>
    val Array(chr, pos) = x.split(":")
    Locus(chr, pos.toInt)
  },
    """
    Construct a :ref:`locus` object.

    .. code-block:: text
        :emphasize-lines: 2

        let l = Locus("1:10040532") in l.position
        result: 10040532
    """,
    ("s", "String of the form ``CHR:POS``")
  )

  register("Locus", { (x: String, y: Int) => Locus(x, y) },
    """
    Construct a :ref:`locus` object.

    .. code-block:: text
        :emphasize-lines: 2

        let l = Locus("1", 10040532) in l.position
        result: 10040532
    """,
    "contig" -> "String representation of contig.",
    "pos" -> "SNP position or start of an indel.")
  register("Interval", { (x: Locus, y: Locus) => Interval(x, y) },
    """
    Construct a :ref:`interval` object. Intervals are **left inclusive, right exclusive**.  This means that ``[chr1:1, chr1:3)`` contains ``chr1:1`` and ``chr1:2``.
    """,
    "startLocus" -> "Start position of interval",
    "endLocus" -> "End position of interval")

  val hweStruct = TStruct(Array(("rExpectedHetFrequency", TFloat64, "Expected rHeterozygosity based on Hardy Weinberg Equilibrium"),
    ("pHWE", TFloat64, "P-value")).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })

  registerAnn("hwe", hweStruct, { (nHomRef: Int, nHet: Int, nHomVar: Int) =>
    if (nHomRef < 0 || nHet < 0 || nHomVar < 0)
      fatal(s"got invalid (negative) argument to function `hwe': hwe($nHomRef, $nHet, $nHomVar)")
    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    Annotation(divOption(LH.getNumericalMean, n).orNull, LH.exactMidP(nAB))
  },
    """
    Compute Hardy Weinberg Equilbrium (HWE) p-value.

    **Examples**

    Compute HWE p-value per variant:

    >>> (vds.annotate_variants_expr('va.hwe = '
    ...     'let nHomRef = gs.filter(g => g.isHomRef()).count().toInt32() and '
    ...     'nHet = gs.filter(g => g.isHet()).count().toInt32() and '
    ...     'nHomVar = gs.filter(g => g.isHomVar()).count().toInt32() in '
    ...     'hwe(nHomRef, nHet, nHomVar)'))

    **Notes**

    Hail computes the exact p-value with mid-p-value correction, i.e. the probability of a less-likely outcome plus one-half the probability of an equally-likely outcome. See this `document <../LeveneHaldane.pdf>`_ for details on the Levene-Haldane distribution and references.
    """,
    "nHomRef" -> "Number of samples that are homozygous for the reference allele.",
    "nHet" -> "Number of samples that are heterozygotes.",
    "nHomVar" -> "Number of samples that are homozygous for the alternate allele."
  )


  def chisqTest(c1: Int, c2: Int, c3: Int, c4: Int): (Option[Double], Option[Double]) = {

    var or: Option[Double] = None
    var chisqp: Option[Double] = None

    if (Array(c1, c2, c3, c4).sum > 0) {
      if (c1 > 0 && c3 == 0)
        or = Option(Double.PositiveInfinity)
      else if (c3 > 0 && c1 == 0)
        or = Option(Double.NegativeInfinity)
      else if ((c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0))
        or = Option((c1 * c4) / (c2 * c3).toDouble)
      chisqp = Option(chisq.chiSquareTest(Array(Array(c1, c2), Array(c3, c4))))
    }

    (chisqp, or)
  }


  val chisqStruct = TStruct(Array(("pValue", TFloat64, "p-value"), ("oddsRatio", TFloat64, "odds ratio")).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })
  registerAnn("chisq", chisqStruct, { (c1: Int, c2: Int, c3: Int, c4: Int) =>
    if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0)
      fatal(s"got invalid argument to function `chisq': chisq($c1, $c2, $c3, $c4)")

    val res = chisqTest(c1, c2, c3, c4)
    Annotation(res._1.orNull, res._2.orNull)
  },
    """ Calculates p-value (Chi-square approximation) and odds ratio for 2x2 table """,
    "c1" -> "value for cell 1", "c2" -> "value for cell 2", "c3" -> "value for cell 3", "c4" -> "value for cell 4"
  )

  registerAnn("ctt", chisqStruct, { (c1: Int, c2: Int, c3: Int, c4: Int, minCellCount: Int) =>
    if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0)
      fatal(s"got invalid argument to function `ctTest': ctTest($c1, $c2, $c3, $c4)")

    if (Array(c1, c2, c3, c4).exists(_ < minCellCount)) {
      val fet = FisherExactTest(c1, c2, c3, c4)
      Annotation(fet(0).orNull, fet(1).orNull)
    } else {
      val res = chisqTest(c1, c2, c3, c4)
      Annotation(res._1.orNull, res._2.orNull)
    }

  },
    """ Calculates p-value and odds ratio for 2x2 table. If any cell is lower than `minCellCount` Fishers exact test is used, otherwise faster chi-squared approximation is used. """,
    "c1" -> "value for cell 1", "c2" -> "value for cell 2", "c3" -> "value for cell 3", "c4" -> "value for cell 4", "minCellCount" -> "Minimum cell count for using chi-squared approximation"
  )

  val fetStruct = TStruct(Array(("pValue", TFloat64, "p-value"), ("oddsRatio", TFloat64, "odds ratio"),
    ("ci95Lower", TFloat64, "lower bound for 95% confidence interval"), ("ci95Upper", TFloat64, "upper bound for 95% confidence interval")).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })


  registerAnn("fet", fetStruct, { (c1: Int, c2: Int, c3: Int, c4: Int) =>
    if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0)
      fatal(s"got invalid argument to function `fet': fet($c1, $c2, $c3, $c4)")
    val fet = FisherExactTest(c1, c2, c3, c4)
    Annotation(fet(0).orNull, fet(1).orNull, fet(2).orNull, fet(3).orNull)
  },
    """
    Calculates the p-value, odds ratio, and 95% confidence interval with Fisher's exact test (FET) for 2x2 tables.

    **Examples**

    Annotate each variant with Fisher's exact test association results (assumes minor/major allele count variant annotations have been computed):

    >>> (vds.annotate_variants_expr(
    ...   'va.fet = let macCase = gs.filter(g => sa.pheno.isCase).map(g => g.nNonRefAlleles()).sum() and '
    ...   'macControl = gs.filter(g => !sa.pheno.isCase).map(g => g.nNonRefAlleles()).sum() and '
    ...   'majCase = gs.filter(g => sa.pheno.isCase).map(g => 2 - g.nNonRefAlleles()).sum() and '
    ...   'majControl = gs.filter(g => !sa.pheno.isCase).map(g => 2 - g.nNonRefAlleles()).sum() in '
    ...   'fet(macCase, macControl, majCase, majControl)'))

    **Notes**

    ``fet`` is identical to the version implemented in `R <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html>`_ with default parameters (two-sided, alpha = 0.05, null hypothesis that the odds ratio equals 1).
    """,
    "a" -> "value for cell 1", "b" -> "value for cell 2", "c" -> "value for cell 3", "d" -> "value for cell 4")
  // NB: merge takes two structs, how do I deal with structs?
  register("exp", { (x: Double) => math.exp(x) },
    """
    Returns Euler's number ``e`` raised to the power of the given value ``x``.
    """,
    "x" -> "the exponent to raise ``e`` to.")
  register("log10", { (x: Double) => math.log10(x) },
    """
    Returns the base 10 logarithm of the given value ``x``.
    """,
    "x" -> "the number to take the base 10 logarithm of.")
  register("sqrt", { (x: Double) => math.sqrt(x) },
    """
    Returns the square root of the given value ``x``.
    """,
    "x" -> "the number to take the square root of.")
  register("log", (x: Double) => math.log(x),
    """
    Returns the natural logarithm of the given value ``x``.
    """,
    "x" -> "the number to take the natural logarithm of.")
  register("log", (x: Double, b: Double) => math.log(x) / math.log(b),
    """
    Returns the base ``b`` logarithm of the given value ``x``.
    """,
    "x" -> "the number to take the base ``b`` logarithm of.",
    "b" -> "the base.")
  register("pow", (b: Double, x: Double) => math.pow(b, x),
    """
    Returns ``b`` raised to the power of ``x``.
    """,
    "b" -> "the base.",
    "x" -> "the exponent.")

  register("Interval", (s: String) => Locus.parseInterval(s),
    """
    Returns an interval parsed in the same way as :py:meth:`~hail.representation.Interval.parse`
    """,
    "s" -> "The string to parse."
  )

  register("Interval", (chr: String, start: Int, end: Int) => Interval(Locus(chr, start), Locus(chr, end)),
    """
    Constructs an interval from a given chromosome, start, and end.
    """,
    "chr" -> "Chromosome.",
    "start" -> "Starting position.",
    "end" -> "Ending position (exclusive).")

  register("pcoin", { (p: Double) => math.random < p },
    """
    Returns true with probability ``p``. This function is non-deterministic.
    """,
    "p" -> "Probability. Should be between 0.0 and 1.0.")
  register("runif", { (min: Double, max: Double) => min + (max - min) * math.random },
    """
    Returns a random draw from a uniform distribution on [``min``, ``max``). ``min`` should be less than or equal to ``max``. This function is non-deterministic.
    """,
    "min" -> "Minimum value of interval.",
    "max" -> "Maximum value of interval, non-inclusive.")

  register("rnorm", { (mean: Double, sd: Double) => mean + sd * scala.util.Random.nextGaussian() },
    """
    Returns a random draw from a normal distribution with mean ``mean`` and standard deviation ``sd``. ``sd`` should be non-negative. This function is non-deterministic.
    """,
    "mean" -> "Mean value of normal distribution.",
    "sd" -> "Standard deviation of normal distribution.")
  register("pnorm", { (x: Double) => pnorm(x) },
    """
    Returns left-tail probability p for which p = Prob(:math:`Z` < ``x``) with :math:`Z` a standard normal random variable.
    """,
    "x" -> "Number at which to compute the probability.")
  register("qnorm", { (p: Double) => qnorm(p) },
    """
    Returns left-quantile x for which p = Prob(:math:`Z` < x) with :math:`Z` a standard normal random variable. ``p`` must satisfy 0 < ``p`` < 1. Inverse of pnorm.
    """,
    "p" -> "Probability")

  register("rpois", { (lambda: Double) => rpois(lambda) },
    """
    Returns a random draw from a Poisson distribution with rate parameter ``lambda``. This function is non-deterministic.
    """,
    "lambda" -> "Poisson rate parameter. Must be non-negative.")
  register("rpois", { (n: Int, lambda: Double) => rpois(n, lambda) },
    """
    Returns ``n`` random draws from a Poisson distribution with rate parameter ``lambda``. This function is non-deterministic.
    """,
    "n" -> "Number of random draws to make.",
    "lambda" -> "Poisson rate parameter. Must be non-negative.")(int32Hr, float64Hr, arrayHr(float64Hr))
  register("dpois", { (x: Double, lambda: Double) => dpois(x, lambda) },
    """
    Returns Prob(:math:`X` = ``x``) from a Poisson distribution with rate parameter ``lambda``.
    """,
    "x" -> "Non-negative number at which to compute the probability density.",
    "lambda" -> "Poisson rate parameter. Must be non-negative.")
  register("dpois", { (x: Double, lambda: Double, logP: Boolean) => dpois(x, lambda, logP) },
    """
    Returns Prob(:math:`X` = ``x``) from a Poisson distribution with rate parameter ``lambda``.
    """,
    "x" -> "Non-negative number at which to compute the probability density.",
    "lambda" -> "Poisson rate parameter. Must be non-negative.",
    "logP" -> "If true, probabilities are returned as log(p).")
  register("ppois", { (x: Double, lambda: Double) => ppois(x, lambda) },
    """
    Returns the left-tail Prob(:math:`X \leq` ``x``) where :math:`X` is a Poisson random variable with rate parameter ``lambda``.
    """,
    "x" -> "Non-negative bound for the left-tail cumulative probability.",
    "lambda" -> "Poisson rate parameter. Must be non-negative.")
  register("ppois", { (x: Double, lambda: Double, lowerTail: Boolean, logP: Boolean) => ppois(x, lambda, lowerTail, logP) },
    """
    If ``lowerTail`` equals true, returns Prob(:math:`X \leq` ``x``) where :math:`X` is a Poisson random variable with rate parameter ``lambda``. If ``lowerTail`` equals false, returns Prob(:math:`X` > ``x``).
    """,
    "x" -> "Non-negative number at which to compute the probability density.",
    "lambda" -> "Poisson rate parameter. Must be non-negative.",
    "lowerTail" -> "If false, returns the exclusive right-tail probability :math:`P(X > x)`.",
    "logP" -> "If true, probabilities are returned as log(p).")

  register("qpois", { (p: Double, lambda: Double) => qpois(p, lambda) },
    """
    Returns the smallest integer :math:`x` such that Prob(:math:`X \leq x`) :math:`\geq` ``p`` where :math:`X` is a Poisson random variable with rate parameter ``lambda``. Inverts ppois.
    """,
    "p" -> """Quantile to compute. Must satisfy :math:`0 \leq p \leq 1`.""",
    "lambda" -> "Poisson rate parameter. Must be non-negative.")

  register("qpois", { (p: Double, lambda: Double, lowerTail: Boolean, logP: Boolean) => qpois(p, lambda, lowerTail, logP) },
    """
    If ``lowerTail`` equals true, returns the smallest integer :math:`x` such that Prob(:math:`X \leq x`) :math:`\geq` ``p`` where :math:`X` is a Poisson random variable with rate parameter ``lambda``.
    If ``lowerTail`` equals false, returns the largest integer :math:`x` such that Prob(:math:`X > x`) :math:`\geq` ``p``. Inverts ppois.
    """,
    "p" -> """Quantile to compute. Must satisfy :math:`0 \leq p \leq 1`.""",
    "lambda" -> "Poisson rate parameter. Must be non-negative.",
    "lowerTail" -> """If false, returns the right-tail inverse cumulative density function.""",
    "logP" -> "If true, input quantiles are given as log(p).")

  register("pchisqtail", { (x: Double, df: Double) => chiSquaredTail(df, x) },
    """
    Returns right-tail probability p for which p = Prob(:math:`Z^2` > x) with :math:`Z^2` a chi-squared random variable with degrees of freedom specified by ``df``. ``x`` must be positive.
    """,
    "x" -> "Number at which to compute the probability.", "df" -> "Degrees of freedom.")
  register("qchisqtail", { (p: Double, df: Double) => inverseChiSquaredTail(df, p) },
    """
    Returns right-quantile x for which p = Prob(:math:`Z^2` > x) with :math:`Z^2` a chi-squared random variable with degrees of freedom specified by ``df``. ``p`` must satisfy 0 < p <= 1. Inverse of pchisq1tail.
    """,
    "p" -> "Probability", "df" -> "Degrees of freedom.")

  register("!", (a: Boolean) => !a, "Negates a boolean variable.")

  def iToD(x: Code[java.lang.Integer]): Code[java.lang.Double] =
    Code.boxDouble(Code.intValue(x).toD)

  def lToD(x: Code[java.lang.Long]): Code[java.lang.Double] =
    Code.boxDouble(Code.longValue(x).toD)

  def iToL(x: Code[java.lang.Integer]): Code[java.lang.Long] =
    Code.boxLong(Code.intValue(x).toL)

  def fToD(x: Code[java.lang.Float]): Code[java.lang.Double] =
    Code.boxDouble(Code.floatValue(x).toD)

  registerConversion((x: java.lang.Integer) => x.toDouble: java.lang.Double, (iToD _).andThen(CM.ret _), priority = 2)
  registerConversion((x: java.lang.Long) => x.toDouble: java.lang.Double, (lToD _).andThen(CM.ret _))
  registerConversion((x: java.lang.Integer) => x.toLong: java.lang.Long, (iToL _).andThen(CM.ret _))
  registerConversion((x: java.lang.Float) => x.toDouble: java.lang.Double, (fToD _).andThen(CM.ret _))

  registerConversion((x: IndexedSeq[java.lang.Integer]) => x.map { xi =>
    if (xi == null)
      null
    else
      box(xi.toDouble)
  }, { (x: Code[IndexedSeq[java.lang.Integer]]) =>
    CM.mapIS(x, (xi: Code[java.lang.Integer]) => xi.mapNull(iToD _))
  }, priority = 2)(arrayHr(boxedInt32Hr), arrayHr(boxedFloat64Hr))

  registerConversion((x: IndexedSeq[java.lang.Long]) => x.map { xi =>
    if (xi == null)
      null
    else
      box(xi.toDouble)
  }, { (x: Code[IndexedSeq[java.lang.Long]]) =>
    CM.mapIS(x, (xi: Code[java.lang.Long]) => xi.mapNull(lToD _))
  })(arrayHr(boxedInt64Hr), arrayHr(boxedFloat64Hr))

  registerConversion((x: IndexedSeq[java.lang.Integer]) => x.map { xi =>
    if (xi == null)
      null
    else
      box(xi.asInstanceOf[Int].toLong)
  }, { (x: Code[IndexedSeq[java.lang.Integer]]) =>
    CM.mapIS(x, (xi: Code[java.lang.Integer]) => xi.mapNull(iToL _))
  })(arrayHr(boxedInt32Hr), arrayHr(boxedInt64Hr))

  registerConversion((x: IndexedSeq[java.lang.Float]) => x.map { xi =>
    if (xi == null)
      null
    else
      box(xi.toDouble)
  }, { (x: Code[IndexedSeq[java.lang.Float]]) =>
    CM.mapIS(x, (xi: Code[java.lang.Float]) => xi.mapNull(fToD _))
  })(arrayHr(boxedFloat32Hr), arrayHr(boxedFloat64Hr))

  register("gtj", (i: Int) => Genotype.gtPair(i).j,
    """
    Convert from genotype index (triangular numbers) to ``j/k`` pairs. Returns ``j``.
    """,
    "i" -> "Genotype index.")
  register("gtk", (i: Int) => Genotype.gtPair(i).k,
    """
    Convert from genotype index (triangular numbers) to ``j/k`` pairs. Returns ``k``.
    """,
    "k" -> "Genotype index.")
  register("gtIndex", (j: Int, k: Int) => Genotype.gtIndexWithSwap(j, k),
    """
    Convert from ``j/k`` pair to genotype index (triangular numbers).
    """,
    "j" -> "j in ``j/k`` pairs.",
    "k" -> "k in ``j/k`` pairs.")

  registerConversion({ (x: java.lang.Integer) =>
    if (x != null)
      box(x.toDouble)
    else
      null
  }, { (x: Code[java.lang.Integer]) => x.mapNull(iToD _)
  }, priority = 2)(aggregableHr(boxedInt32Hr), aggregableHr(boxedFloat64Hr))

  registerConversion({ (x: java.lang.Long) =>
    if (x != null)
      box(x.toDouble)
    else
      null
  }, { (x: Code[java.lang.Long]) => x.mapNull(lToD _)
  })(aggregableHr(boxedInt64Hr), aggregableHr(boxedFloat64Hr))

  registerConversion({ (x: java.lang.Integer) =>
    if (x != null)
      box(x.toLong)
    else
      null
  }, { (x: Code[java.lang.Integer]) => x.mapNull(iToL _)
  })(aggregableHr(boxedInt32Hr), aggregableHr(boxedInt64Hr))

  registerConversion({ (x: java.lang.Float) =>
    if (x != null)
      box(x.toDouble)
    else
      null
  }, { (x: Code[java.lang.Float]) => x.mapNull(fToD _)
  })(aggregableHr(boxedFloat32Hr), aggregableHr(boxedFloat64Hr))

  registerMethod("split", (s: String, p: String) => s.split(p): IndexedSeq[String],
    """
    Returns an array of strings, split on the given regular expression delimiter. See the documentation on `regular expression syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_ delimiter. If you need to split on special characters, escape them with double backslash (\\\\).

    .. code-block:: text
        :emphasize-lines: 2

        let s = "1kg-NA12878" in s.split("-")
        result: ["1kg", "NA12878"]
    """,
    "delim" -> "Regular expression delimiter.")

  registerMethod("split", (s: String, p: String, n: Int) => s.split(p, n): IndexedSeq[String],
    """
    Returns an array of strings, split on the given regular expression delimiter with the pattern applied ``n`` times. See the documentation on `regular expression syntax <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_ delimiter. If you need to split on special characters, escape them with double backslash (\\\\).

    .. code-block:: text
        :emphasize-lines: 2

        let s = "1kg-NA12878" in s.split("-")
        result: ["1kg", "NA12878"]
    """,
    "delim" -> "Regular expression delimiter.", "n" -> "Number of times the pattern is applied. See the `Java documentation <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#split-java.lang.String-int->`_ for more information."
  )

  registerMethod("oneHotAlleles", (g: Genotype, v: Variant) => Genotype.oneHotAlleles(v, g).orNull,
    """
    Produce an array of called counts for each allele in the variant (including reference). For example, calling this function with a biallelic variant on hom-ref, het, and hom-var genotypes will produce ``[2, 0]``, ``[1, 1]``, and ``[0, 2]`` respectively.
    """,
    "v" -> ":ref:`variant`")

  registerMethod("oneHotGenotype", (g: Genotype, v: Variant) => Genotype.oneHotGenotype(v, g).orNull,
    """
    Produces an array with one element for each possible genotype in the variant, where the called genotype is 1 and all else 0. For example, calling this function with a biallelic variant on hom-ref, het, and hom-var genotypes will produce ``[1, 0, 0]``, ``[0, 1, 0]``, and ``[0, 0, 1]`` respectively.
    """,
    "v" -> ":ref:`variant`"
  )

  registerMethod("replace", (str: String, pattern1: String, pattern2: String) =>
    str.replaceAll(pattern1, pattern2),
    """
    Replaces each substring of this string that matches the given `regular expression <https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html>`_ (``pattern1``) with the given replacement (``pattern2``).

    .. code-block:: text
        :emphasize-lines: 2

        let s = "1kg-NA12878" in a.replace("1kg-", "")
        result: "NA12878"
    """,
    "pattern1" -> "Substring to replace.",
    "pattern2" -> "Replacement string.")

  registerMethod("contains", (interval: Interval[Locus], locus: Locus) => interval.contains(locus),
    """
    Returns true if the ``locus`` is in the interval.

    .. code-block:: text
        :emphasize-lines: 2

        let i = Interval(Locus("1", 1000), Locus("1", 2000)) in i.contains(Locus("1", 1500))
        result: true
    """,
    "locus" -> ":ref:`locus`")

  val sizeDocstring = "Number of elements in the collection."
  registerMethod("length", (a: IndexedSeq[Any]) => a.length, sizeDocstring)(arrayHr(TTHr), int32Hr)
  registerMethod("size", (a: IndexedSeq[Any]) => a.size, sizeDocstring)(arrayHr(TTHr), int32Hr)
  registerMethod("size", (s: Set[Any]) => s.size, sizeDocstring)(setHr(TTHr), int32Hr)
  registerMethod("size", (d: Map[Any, Any]) => d.size, sizeDocstring)(dictHr(TTHr, TUHr), int32Hr)

  val isEmptyDocstring = "Returns true if the number of elements is equal to 0. false otherwise."
  registerMethod("isEmpty", (a: IndexedSeq[Any]) => a.isEmpty, isEmptyDocstring)(arrayHr(TTHr), boolHr)
  registerMethod("isEmpty", (s: Set[Any]) => s.isEmpty, isEmptyDocstring)(setHr(TTHr), boolHr)
  registerMethod("isEmpty", (d: Map[Any, Any]) => d.isEmpty, isEmptyDocstring)(dictHr(TTHr, TUHr), boolHr)

  registerMethod("toSet", (a: IndexedSeq[Any]) => a.toSet, "Convert collection to a Set.")(arrayHr(TTHr), setHr(TTHr))
  registerMethod("toSet", (a: Set[Any]) => a, "Convert collection to a Set.")(setHr(TTHr), setHr(TTHr))
  registerMethod("toArray", (a: Set[Any]) => a.toArray[Any]: IndexedSeq[Any], "Convert collection to an Array.")(setHr(TTHr), arrayHr(TTHr))
  registerMethod("toArray", (a: IndexedSeq[Any]) => a, "Convert collection to an Array.")(arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("head", (a: IndexedSeq[Any]) => a.head, "Selects the first element.")(arrayHr(TTHr), TTHr)
  registerMethod("tail", (a: IndexedSeq[Any]) => a.tail, "Selects all elements except the first.")(arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("head", (a: Set[Any]) => a.head, "Select one element.")(setHr(TTHr), TTHr)
  registerMethod("tail", (a: Set[Any]) => a.tail, "Select all elements except the element returned by ``head``.")(setHr(TTHr), setHr(TTHr))

  registerMethod("append", (x: IndexedSeq[Any], a: Any) => x :+ a, "Returns the result of adding the element `a` to the end of this Array.")(arrayHr(TTHr), TTHr, arrayHr(TTHr))
  registerMethod("extend", (x: IndexedSeq[Any], a: IndexedSeq[Any]) => x ++ a, "Returns the concatenation of this Array followed by Array `a`.")(arrayHr(TTHr), arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("add", (x: Set[Any], a: Any) => x + a, "Returns the result of adding the element `a` to this Set.")(setHr(TTHr), TTHr, setHr(TTHr))
  registerMethod("union", (x: Set[Any], a: Set[Any]) => x ++ a, "Returns the union of this Set and Set `a`.")(setHr(TTHr), setHr(TTHr), setHr(TTHr))
  registerMethod("intersection", (x: Set[Any], a: Set[Any]) => x & a, "Returns the intersection of this Set and Set `a`.")(setHr(TTHr), setHr(TTHr), setHr(TTHr))
  registerMethod("difference", (x: Set[Any], a: Set[Any]) => x &~ a, "Returns the elements of this Set that are not in Set `a`.")(setHr(TTHr), setHr(TTHr), setHr(TTHr))
  registerMethod("issubset", (x: Set[Any], a: Set[Any]) => x.subsetOf(a), "Returns true if this Set is a subset of Set `a`.")(setHr(TTHr), setHr(TTHr), boolHr)

  registerMethod("flatten", (a: IndexedSeq[IndexedSeq[Any]]) =>
    flattenOrNull[IndexedSeq, Any](IndexedSeq.newBuilder[Any], a),
    """
    Flattens a nested array by concatenating all its rows into a single array.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [[1, 3], [2, 4]] in a.flatten()
        result: [1, 3, 2, 4]
    """
  )(arrayHr(arrayHr(TTHr)), arrayHr(TTHr))

  registerMethod("flatten", (s: Set[Set[Any]]) =>
    flattenOrNull[Set, Any](Set.newBuilder[Any], s),
    """
    Flattens a nested set by concatenating all its elements into a single set.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [[1, 2].toSet(), [3, 4].toSet()].toSet() in s.flatten()
        result: Set(1, 2, 3, 4)
    """
  )(setHr(setHr(TTHr)), setHr(TTHr))

  registerMethod("keys", (m: Map[Any, Any]) =>
    m.keysIterator.toArray[Any]: IndexedSeq[Any], "Returns an Array containing the keys of the Dict."
  )(dictHr(TTHr, TUHr), arrayHr(TTHr))

  registerMethod("values", (m: Map[Any, Any]) =>
    m.valuesIterator.toArray[Any]: IndexedSeq[Any], "Returns an Array containing the values of the Dict."
  )(dictHr(TTHr, TUHr), arrayHr(TUHr))

  registerMethod("keySet", (m: Map[Any, Any]) =>
    m.keySet, "Returns a Set containing the keys of the Dict."
  )(dictHr(TTHr, TUHr), setHr(TTHr))

  registerMethod("get", (m: Map[Any, Any], key: Any) =>
    m.get(key).orNull, "Returns the value of the Dict for key ``k``, or returns ``NA`` if the key is not found."
  )(dictHr(TTHr, TUHr), TTHr, TUHr)

  registerMethod("mkString", (a: IndexedSeq[String], d: String) => a.mkString(d),
    """
    Concatenates all elements of this array into a single string where each element is separated by the ``delimiter``.

    .. code-block:: text
        :emphasize-lines: 2

        let a = ["a", "b", "c"] in a.mkString("::")
        result: "a::b::c"

    """,
    "delimiter" -> "String that separates each element.")(
    arrayHr(stringHr), stringHr, stringHr)
  registerMethod("mkString", (s: Set[String], d: String) => s.mkString(d),
    """
    Concatenates all elements of this set into a single string where each element is separated by the ``delimiter``.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [1, 2, 3].toSet() in s.mkString(",")
        result: "1,2,3"
    """,
    "delimiter" -> "String that separates each element.")(
    setHr(stringHr), stringHr, stringHr)

  registerMethod("contains", (s: Set[Any], x: Any) => s.contains(x),
    """
    Returns true if the element ``x`` is contained in the set, otherwise false.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [1, 2, 3].toSet() in s.contains(5)
        result: false
    """,
    "x" -> "Value to test."
  )(setHr(TTHr), TTHr, boolHr)
  registerMethod("contains", (d: Map[Any, Any], x: Any) => d.contains(x),
    """
    Returns true if the Dict has a key equal to ``k``, otherwise false.
    """,
    "k" -> "Key name to query."
  )(dictHr(TTHr, TUHr), TTHr, boolHr)

  registerLambdaMethod("find", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.find { elt =>
      val r = f(elt)
      r != null && r.asInstanceOf[Boolean]
    }.orNull,
    """
    Returns the first non-missing element of the array for which expr is true. If no element satisfies the predicate, find returns NA.

    .. code-block:: text
        :emphasize-lines: 2

        let a = ["cat", "dog", "rabbit"] in a.find(e => 'bb' ~ e)
        result: "rabbit"
    """,
    "expr" -> "Lambda expression."
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), TTHr)

  registerLambdaMethod("find", (s: Set[Any], f: (Any) => Any) =>
    s.find { elt =>
      val r = f(elt)
      r != null && r.asInstanceOf[Boolean]
    }.orNull,
    """
    Returns the first non-missing element of the array for which expr is true. If no element satisfies the predicate, find returns NA.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [1, 2, 3].toSet() in s.find(e => e % 3 == 0)
        result: 3
    """,
    "expr" -> "Lambda expression."
  )(setHr(TTHr), unaryHr(TTHr, boolHr), TTHr)

  registerLambdaMethod("map", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.map(f),
    """
    Returns a new array produced by applying ``expr`` to each element.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 1, 2, 3] in a.map(e => pow(2, e))
        result: [1, 2, 4, 8]
    """,
    "expr" -> "Lambda expression."
  )(arrayHr(TTHr), unaryHr(TTHr, TUHr), arrayHr(TUHr))

  registerLambdaMethod("map", (s: Set[Any], f: (Any) => Any) =>
    s.map(f),
    """
    Returns a new set produced by applying ``expr`` to each element.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [1, 2, 3].toSet() in s.map(e => e * 3)
        result: Set(3, 6, 9)
    """,
    "expr" -> "Lambda expression."
  )(setHr(TTHr), unaryHr(TTHr, TUHr), setHr(TUHr))

  registerLambdaMethod("mapValues", (a: Map[Any, Any], f: (Any) => Any) =>
    a.map { case (k, v) => (k, f(v)) },
    """
    Returns a new Dict produced by applying ``expr`` to each value. The keys are unmodified.
    """,
    "expr" -> "Lambda expression."
  )(dictHr[Any, Any](TTHr, TUHr), unaryHr(TUHr, TVHr), dictHr[Any, Any](TTHr, TVHr))

  //  registerMapLambdaMethod("mapValues", (a: Map[Any, Any], f: (Any) => Any) =>
  //    a.map { case (k, v) => (k, f(v)) }
  //  )(TTHr, TUHr, unaryHr(TUHr, TVHr), TVHr)

  registerLambdaMethod("flatMap", (a: IndexedSeq[Any], f: (Any) => Any) =>
    flattenOrNull[IndexedSeq, Any](IndexedSeq.newBuilder[Any],
      a.map(f).asInstanceOf[IndexedSeq[IndexedSeq[Any]]]),
    """
    Returns a new array by applying a function to each subarray and concatenating the resulting arrays.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [[1, 2, 3], [4, 5], [6]] in a.flatMap(e => e + 1)
        result: [2, 3, 4, 5, 6, 7]
    """,
    "expr" -> "Lambda expression."
  )(arrayHr(TTHr), unaryHr(TTHr, arrayHr(TUHr)), arrayHr(TUHr))

  registerLambdaMethod("flatMap", (s: Set[Any], f: (Any) => Any) =>
    flattenOrNull[Set, Any](Set.newBuilder[Any],
      s.map(f).asInstanceOf[Set[Set[Any]]]),
    """
    Returns a new set by applying a function to each subset and concatenating the resulting sets.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [["a", "b", "c"].toSet(), ["d", "e"].toSet(), ["f"].toSet()].toSet() in s.flatMap(e => e + "1")
        result: Set("a1", "b1", "c1", "d1", "e1", "f1")
    """,
    "expr" -> "Lambda expression."
  )(setHr(TTHr), unaryHr(TTHr, setHr(TUHr)), setHr(TUHr))

  registerLambdaMethod("groupBy", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.groupBy(f), null
  )(arrayHr(TTHr), unaryHr(TTHr, TUHr), dictHr(TUHr, arrayHr(TTHr)))

  registerLambdaMethod("groupBy", (a: Set[Any], f: (Any) => Any) =>
    a.groupBy(f), null
  )(setHr(TTHr), unaryHr(TTHr, TUHr), dictHr(TUHr, setHr(TTHr)))

  registerLambdaMethod("exists", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.exists { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    },
    """
    Returns a boolean which is true if **any** element in the array satisfies the condition given by ``expr``. false otherwise.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [1, 2, 3, 4, 5, 6] in a.exists(e => e > 4)
        result: true
    """,
    "expr" -> "Lambda expression."
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("exists", (s: Set[Any], f: (Any) => Any) =>
    s.exists { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    },
    """
    Returns a boolean which is true if **any** element in the set satisfies the condition given by ``expr`` and false otherwise.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [0, 2, 4, 6, 8, 10].toSet() in s.exists(e => e % 2 == 1)
        result: false
    """,
    "expr" -> "Lambda expression."
  )(setHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("forall", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.forall { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    },
    """
    Returns a boolean which is true if **all** elements in the array satisfies the condition given by ``expr`` and false otherwise.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a.forall(e => e % 2 == 0)
        result: true
    """,
    "expr" -> "Lambda expression."
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("forall", (s: Set[Any], f: (Any) => Any) =>
    s.forall { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    },
    """
    Returns a boolean which is true if **all** elements in the set satisfies the condition given by ``expr`` and false otherwise.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [0.1, 0.5, 0.3, 1.0, 2.5, 3.0].toSet() in s.forall(e => e > 1.0 == 0)
        result: false
    """,
    "expr" -> "Lambda expression."
  )(setHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("filter", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.filter { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    },
    """
    Returns a new array subsetted to the elements where ``expr`` evaluates to true.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [1, 4, 5, 6, 10] in a.filter(e => e % 2 == 0)
        result: [4, 6, 10]
    """,
    "expr" -> "Lambda expression."
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), arrayHr(TTHr))

  registerLambdaMethod("filter", (s: Set[Any], f: (Any) => Any) =>
    s.filter { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    },
    """
    Returns a new set subsetted to the elements where ``expr`` evaluates to true.

    .. code-block:: text
        :emphasize-lines: 2

        let s = [1, 4, 5, 6, 10].toSet() in s.filter(e => e >= 5)
        result: Set(5, 6, 10)
    """,
    "expr" -> "Lambda expression."
  )(setHr(TTHr), unaryHr(TTHr, boolHr), setHr(TTHr))

  registerAggregator[Any, Long]("count", () => new CountAggregator(),
    """
    Counts the number of elements in an aggregable.

    **Examples**

    Count the number of heterozygote genotype calls in an aggregable of genotypes (``gs``):

    >>> vds_result = vds.annotate_variants_expr('va.nHets = gs.filter(g => g.isHet()).count()')
    """
  )(aggregableHr(TTHr), int64Hr)

  registerAggregator[Any, IndexedSeq[Any]]("collect", () => new CollectAggregator(),
    """
    Returns an array with all of the elements in the aggregable. Order is not guaranteed.

    **Examples**

    Collect the list of sample IDs with heterozygote genotype calls per variant:

    >>> vds_result = vds.annotate_variants_expr('va.hetSamples = gs.filter(g => g.isHet()).map(g => s).collect()')

    ``va.hetSamples`` will have the type ``Array[String]``.
    """
  )(aggregableHr(TTHr), arrayHr(TTHr))

  registerAggregator[Any, Set[Any]]("collectAsSet", () => new CollectSetAggregator(),
    """
    Returns the set of all unique elements in the aggregable.
    """
  )(aggregableHr(TTHr), setHr(TTHr))

  val sumAggDocstring = """Compute the sum of all non-missing elements. The empty sum is zero."""
  registerAggregator[Int, Int]("sum", () => new SumAggregator[Int](), sumAggDocstring)(aggregableHr(int32Hr), int32Hr)

  registerAggregator[Long, Long]("sum", () => new SumAggregator[Long](), sumAggDocstring)(aggregableHr(int64Hr), int64Hr)

  registerAggregator[Float, Float]("sum", () => new SumAggregator[Float](), sumAggDocstring)(aggregableHr(float32Hr), float32Hr)

  registerAggregator[Double, Double]("sum", () => new SumAggregator[Double](), sumAggDocstring)(aggregableHr(float64Hr), float64Hr)

  registerAggregator[IndexedSeq[Int], IndexedSeq[Int]]("sum", () => new SumArrayAggregator[Int](),
    """
    Compute the sum by index. All elements in the aggregable must have the same length.

    **Examples**

    Count the total number of occurrences of each allele across samples, per variant:

    >>> vds_result = vds.annotate_variants_expr('va.AC = gs.map(g => g.oneHotAlleles(v)).sum()')
    """
  )(aggregableHr(arrayHr(int32Hr)), arrayHr(int32Hr))

  registerAggregator[IndexedSeq[Long], IndexedSeq[Long]]("sum", () => new SumArrayAggregator[Long](),
    "Compute the sum by index. All elements in the aggregable must have the same length."
  )(aggregableHr(arrayHr(int64Hr)), arrayHr(int64Hr))

  registerAggregator[IndexedSeq[Float], IndexedSeq[Float]]("sum", () => new SumArrayAggregator[Float](),
    "Compute the sum by index. All elements in the aggregable must have the same length."
  )(aggregableHr(arrayHr(float32Hr)), arrayHr(float32Hr))

  registerAggregator[IndexedSeq[Double], IndexedSeq[Double]]("sum", () => new SumArrayAggregator[Double](),
    "Compute the sum by index. All elements in the aggregable must have the same length."
  )(aggregableHr(arrayHr(float64Hr)), arrayHr(float64Hr))

  val productAggDocstring = """Compute the product of all non-missing elements. The empty product is one."""
  registerAggregator[Long, Long]("product", () => new ProductAggregator[Long](), productAggDocstring)(aggregableHr(int64Hr), int64Hr)

  registerAggregator[Double, Double]("product", () => new ProductAggregator[Double](), productAggDocstring)(aggregableHr(float64Hr), float64Hr)

  val maxAggDocstring = """Compute the maximum of all non-missing elements. The empty max is missing."""
  registerAggregator[Int, java.lang.Integer]("max", () => new MaxAggregator[Int, java.lang.Integer](), maxAggDocstring)(aggregableHr(int32Hr), boxedInt32Hr)

  registerAggregator[Long, java.lang.Long]("max", () => new MaxAggregator[Long, java.lang.Long](), maxAggDocstring)(aggregableHr(int64Hr), boxedInt64Hr)

  registerAggregator[Float, java.lang.Float]("max", () => new MaxAggregator[Float, java.lang.Float](), maxAggDocstring)(aggregableHr(float32Hr), boxedFloat32Hr)

  registerAggregator[Double, java.lang.Double]("max", () => new MaxAggregator[Double, java.lang.Double](), maxAggDocstring)(aggregableHr(float64Hr), boxedFloat64Hr)

  val minAggDocstring = """Compute the minimum of all non-missing elements. The empty min is missing."""
  registerAggregator[Int, java.lang.Integer]("min", () => new MinAggregator[Int, java.lang.Integer](), minAggDocstring)(aggregableHr(int32Hr), boxedInt32Hr)

  registerAggregator[Long, java.lang.Long]("min", () => new MinAggregator[Long, java.lang.Long](), minAggDocstring)(aggregableHr(int64Hr), boxedInt64Hr)

  registerAggregator[Float, java.lang.Float]("min", () => new MinAggregator[Float, java.lang.Float](), minAggDocstring)(aggregableHr(float32Hr), boxedFloat32Hr)

  registerAggregator[Double, java.lang.Double]("min", () => new MinAggregator[Double, java.lang.Double](), minAggDocstring)(aggregableHr(float64Hr), boxedFloat64Hr)

  registerAggregator[IndexedSeq[Double], Any]("infoScore", () => new InfoScoreAggregator(),
    """
    Compute the IMPUTE information score.

    **Examples**

    Calculate the info score per variant:

    >>> (hc.import_gen("data/example.gen", "data/example.sample")
    ...    .annotate_variants_expr('va.infoScore = gs.map(g => g.GP).infoScore()'))

    Calculate group-specific info scores per variant:

    >>> vds_result = (hc.import_gen("data/example.gen", "data/example.sample")
    ...    .annotate_samples_expr("sa.isCase = pcoin(0.5)")
    ...    .annotate_variants_expr(["va.infoScore.case = gs.filter(g => sa.isCase).map(g => g.GP).infoScore()",
    ...                             "va.infoScore.control = gs.filter(g => !sa.isCase).map(g => g.GP).infoScore()"]))

    **Notes**

    We implemented the IMPUTE info measure as described in the `supplementary information from Marchini & Howie. Genotype imputation for genome-wide association studies. Nature Reviews Genetics (2010) <http://www.nature.com/nrg/journal/v11/n7/extref/nrg2796-s3.pdf>`_.

    To calculate the info score :math:`I_{A}` for one SNP:

    .. math::

        I_{A} =
        \begin{cases}
        1 - \frac{\sum_{i=1}^{N}(f_{i} - e_{i}^2)}{2N\hat{\theta}(1 - \hat{\theta})} & \text{when } \hat{\theta} \in (0, 1) \\
        1 & \text{when } \hat{\theta} = 0, \hat{\theta} = 1\\
        \end{cases}

    - :math:`N` is the number of samples with imputed genotype probabilities [:math:`p_{ik} = P(G_{i} = k)` where :math:`k \in \{0, 1, 2\}`]
    - :math:`e_{i} = p_{i1} + 2p_{i2}` is the expected genotype per sample
    - :math:`f_{i} = p_{i1} + 4p_{i2}`
    - :math:`\hat{\theta} = \frac{\sum_{i=1}^{N}e_{i}}{2N}` is the MLE for the population minor allele frequency


    Hail will not generate identical results as `QCTOOL <http://www.well.ox.ac.uk/~gav/qctool/#overview>`_ for the following reasons:

      - The floating point number Hail stores for each genotype probability is slightly different than the original data due to rounding and normalization of probabilities.
      - Hail automatically removes genotype probability distributions that do not meet certain requirements on data import with :py:meth:`~hail.HailContext.import_gen` and :py:meth:`~hail.HailContext.import_bgen`.
      - Hail does not use the population frequency to impute genotype probabilities when a genotype probability distribution has been set to missing.
      - Hail calculates the same statistic for sex chromosomes as autosomes while QCTOOL incorporates sex information

    .. warning::

        - The info score Hail reports will be extremely different from qctool when a SNP has a high missing rate.
        - It only makes sense to compute the info score per variant. While a per-sample info score will run, the result is meaningless.
    """)(aggregableHr(arrayHr(float64Hr)),
    new HailRep[Any] {
      def typ: Type = InfoScoreCombiner.signature
    })

  registerAggregator[Genotype, Any]("hardyWeinberg", () => new HWEAggregator(),
    """
    Compute Hardy-Weinberg equilibrium p-value.

    **Examples**

    Add a new variant annotation that calculates HWE p-value by phenotype:

    >>> vds_result = vds.annotate_variants_expr([
    ...   'va.hweCase = gs.filter(g => sa.pheno.isCase).hardyWeinberg()',
    ...   'va.hweControl = gs.filter(g => !sa.pheno.isCase).hardyWeinberg()'])

    **Notes**

    Hail computes the exact p-value with mid-p-value correction, i.e. the probability of a less-likely outcome plus one-half the probability of an equally-likely outcome. See this `document <../LeveneHaldane.pdf>`_ for details on the Levene-Haldane distribution and references.
    """
  )(aggregableHr(genotypeHr),
    new HailRep[Any] {
      def typ = HWECombiner.signature
    })

  registerAggregator[Any, Any]("counter", () => new CounterAggregator(),
    """
    Counts the number of occurrences of each element in an aggregable.

    **Examples**

    Compute the number of indels in each chromosome:

    >>> [indels_per_chr] = vds.query_variants(['variants.filter(v => v.altAllele().isIndel()).map(v => v.contig).counter()'])

    **Notes**

    We recommend this function is used with the `Python counter object <https://docs.python.org/2/library/collections.html#collections.Counter>`_.

    >>> [counter] = vds.query_variants(['variants.flatMap(v => v.altAlleles).counter()'])
    >>> from collections import Counter
    >>> counter = Counter(counter)
    >>> print(counter.most_common(5))
    [(AltAllele(C, T), 129L),
     (AltAllele(G, A), 112L),
     (AltAllele(C, A), 60L),
     (AltAllele(A, G), 46L),
     (AltAllele(T, C), 44L)]
    """)(aggregableHr(TTHr),
    new HailRep[Any] {
      def typ = TDict(TTHr.typ, TInt64)
    })

  registerAggregator[Double, Any]("stats", () => new StatAggregator(),
    """
    Compute summary statistics about a numeric aggregable.

    **Examples**

    Compute the mean genotype quality score per variant:

    >>> vds_result = vds.annotate_variants_expr('va.gqMean = gs.map(g => g.gq).stats().mean')

    Compute summary statistics on the number of singleton calls per sample:

    >>> [singleton_stats] = (vds.sample_qc()
    ...     .query_samples(['samples.map(s => sa.qc.nSingleton).stats()']))

    Compute GQ and DP statistics stratified by genotype call:

    >>> gq_dp = [
    ... 'va.homrefGQ = gs.filter(g => g.isHomRef()).map(g => g.gq).stats()',
    ... 'va.hetGQ = gs.filter(g => g.isHet()).map(g => g.gq).stats()',
    ... 'va.homvarGQ = gs.filter(g => g.isHomVar()).map(g => g.gq).stats()',
    ... 'va.homrefDP = gs.filter(g => g.isHomRef()).map(g => g.dp).stats()',
    ... 'va.hetDP = gs.filter(g => g.isHet()).map(g => g.dp).stats()',
    ... 'va.homvarDP = gs.filter(g => g.isHomVar()).map(g => g.dp).stats()']
    >>> vds_result = vds.annotate_variants_expr(gq_dp)

    **Notes**

    The ``stats()`` aggregator can be used to replicate some of the values computed by :py:meth:`~hail.VariantDataset.variant_qc` and :py:meth:`~hail.VariantDataset.sample_qc` such as ``dpMean`` and ``dpStDev``.
    """)(aggregableHr(float64Hr),
    new HailRep[Any] {
      def typ = TStruct(Array(
        ("mean", TFloat64, "Mean value"), ("stdev", TFloat64, "Standard deviation"), ("min", TFloat64, "Minimum value"),
        ("max", TFloat64, "Maximum value"), ("nNotMissing", TInt64, "Number of non-missing values"), ("sum", TFloat64, "Sum of all elements")
      ).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })
    })

  registerAggregator[Double, Double, Double, Int, Any]("hist", (start: Double, end: Double, bins: Int) => {
    if (bins <= 0)
      fatal(s"""method `hist' expects `bins' argument to be > 0, but got $bins""")

    val binSize = (end - start) / bins
    if (binSize <= 0)
      fatal(
        s"""invalid bin size from given arguments (start = $start, end = $end, bins = $bins)
           |  Method requires positive bin size [(end - start) / bins], but got ${ binSize.formatted("%.2f") }
                  """.stripMargin)

    val indices = Array.tabulate(bins + 1)(i => start + i * binSize)

    new HistAggregator(indices)
  },
    """
    Compute frequency distributions of numeric parameters.

    **Examples**

    Compute GQ-distributions per variant:

    >>> vds_result = vds.annotate_variants_expr('va.gqHist = gs.map(g => g.gq).hist(0, 100, 20)')

    Compute global GQ-distribution:

    >>> gq_hist = vds.query_genotypes('gs.map(g => g.gq).hist(0, 100, 100)')

    **Notes**

    - The start, end, and bins params are no-scope parameters, which means that while computations like 100 / 4 are acceptable, variable references like ``global.nBins`` are not.
    - Bin size is calculated from (``end`` - ``start``) / ``bins``
    - (``bins`` + 1) breakpoints are generated from the range (start to end by binsize)
    - Each bin is left-inclusive, right-exclusive except the last bin, which includes the maximum value. This means that if there are N total bins, there will be N + 1 elements in ``binEdges``. For the invocation ``hist(0, 3, 3)``, ``binEdges`` would be ``[0, 1, 2, 3]`` where the bins are ``[0, 1), [1, 2), [2, 3]``.
    """, "start" -> "Starting point of first bin", "end" -> "End point of last bin", "bins" -> "Number of bins to create.")(aggregableHr(float64Hr), float64Hr, float64Hr, int32Hr, new HailRep[Any] {
    def typ = HistogramCombiner.schema
  })

  registerLambdaAggregator[Genotype, (Any) => Any, Any]("callStats", (vf: (Any) => Any) => new CallStatsAggregator(vf),
    """
    Compute four commonly-used metrics over a set of genotypes in a variant.

    **Examples**

    Compute phenotype-specific call statistics:

    >>> pheno_stats = [
    ...   'va.case_stats = gs.filter(g => sa.pheno.isCase).callStats(g => v)',
    ...   'va.control_stats = gs.filter(g => !sa.pheno.isCase).callStats(g => v)']
    >>> vds_result = vds.annotate_variants_expr(pheno_stats)

    ``va.eur_stats.AC`` will be the allele count (AC) computed from individuals marked as "EUR".
    """, "f" -> "Variant lambda expression such as ``g => v``."
  )(
    aggregableHr(genotypeHr), unaryHr(genotypeHr, variantHr), new HailRep[Any] {
      def typ = CallStats.schema
    })

  registerLambdaAggregator[Genotype, (Any) => Any, Any]("inbreeding", (af: (Any) => Any) => new InbreedingAggregator(af),
    """
    Compute inbreeding metric. This aggregator is equivalent to the `\`--het\` method in PLINK <https://www.cog-genomics.org/plink2/basic_stats#ibc>`_.

    **Examples**

    Calculate the inbreeding metric per sample:

    >>> vds_result = (vds.variant_qc()
    ...     .annotate_samples_expr('sa.inbreeding = gs.inbreeding(g => va.qc.AF)'))

    To obtain the same answer as `PLINK <https://www.cog-genomics.org/plink2>`_, use the following series of commands:

    >>> vds_result = (vds.variant_qc()
    ...     .filter_variants_expr('va.qc.AC > 1 && va.qc.AF >= 1e-8 && va.qc.nCalled * 2 - va.qc.AC > 1 && va.qc.AF <= 1 - 1e-8 && v.isAutosomal()')
    ...     .annotate_samples_expr('sa.inbreeding = gs.inbreeding(g => va.qc.AF)'))

    **Notes**

    The Inbreeding Coefficient (F) is computed as follows:

    #. For each variant and sample with a non-missing genotype call, ``E``, the expected number of homozygotes (computed from user-defined expression for minor allele frequency), is computed as ``1.0 - (2.0*maf*(1.0-maf))``
    #. For each variant and sample with a non-missing genotype call, ``O``, the observed number of homozygotes, is computed as ``0 = heterozygote; 1 = homozygote``
    #. For each variant and sample with a non-missing genotype call, ``N`` is incremented by 1
    #. For each sample, ``E``, ``O``, and ``N`` are combined across variants
    #. ``F`` is calculated by ``(O - E) / (N - E)``
    """, "af" -> "Lambda expression for the alternate allele frequency.")(
    aggregableHr(genotypeHr), unaryHr(genotypeHr, float64Hr), new HailRep[Any] {
      def typ = InbreedingCombiner.signature
    })

  registerLambdaAggregator[Any, (Any) => Any, java.lang.Double]("fraction", (f: (Any) => Any) => new FractionAggregator(f),
    """
    Computes the ratio of the number of occurrences for which a boolean condition evaluates to true, divided by the number of included elements in the aggregable.

    **Examples**

    Filter variants with a call rate less than 95%:

    >>> vds_result = vds.filter_variants_expr('gs.fraction(g => g.isCalled()) > 0.90')

    Compute the differential missingness at SNPs and indels:

    >>> exprs = ['sa.SNPmissingness = gs.filter(g => v.altAllele().isSNP()).fraction(g => g.isNotCalled())',
    ...          'sa.indelmissingness = gs.filter(g => v.altAllele().isIndel()).fraction(g => g.isNotCalled())']
    >>> vds_result = vds.annotate_samples_expr(exprs)
    """)(
    aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boxedFloat64Hr)

  registerLambdaAggregator[Any, (Any) => Any, Boolean]("exists", (f: (Any) => Any) => new ExistsAggregator(f),
    """
    Returns true if **any** element in the aggregator satisfies the condition given by ``expr`` and false otherwise.
    """,
    "expr" -> "Lambda expression.")(aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boolHr)

  registerLambdaAggregator[Any, (Any) => Any, Boolean]("forall", (f: (Any) => Any) => new ForallAggregator(f),
    """
    Returns a true if **all** elements in the array satisfies the condition given by ``expr`` and false otherwise.
    """,
    "expr" -> "Lambda expression.")(aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boolHr)

  registerAggregator("take", (n: Int) => new TakeAggregator(n),
    """
    Take the first ``n`` items of an aggregable.

    **Examples**

    Collect the first 5 sample IDs with at least one alternate allele per variant:

    >>> vds_result = vds.annotate_variants_expr("va.nonRefSamples = gs.filter(g => g.nNonRefAlleles() > 0).map(g => s).take(5)")
    """, "n" -> "Number of items to take.")(
    aggregableHr(TTHr), int32Hr, arrayHr(TTHr))

  private val genericTakeByDocs = """
    Returns the first ``n`` items of an aggregable in ascending order, ordered
    by the result of ``f``. ``NA`` always appears last. If the aggregable
    contains less than ``n`` items, then the result will contain as many
    elements as the aggregable contains.

    """

  private val integralTakeByDocs = genericTakeByDocs ++ """
    **Examples**

    Consider an aggregable ``gs`` containing these elements::

      7, 6, 3, NA, 1, 2, NA, 4, 5, -1

    The expression ``gs.takeBy(x => x, 5)`` would return the array::

      [-1, 1, 2, 3, 4]

    The expression ``gs.takeBy(x => -x, 5)`` would return the array::

      [7, 6, 5, 4, 3]

    The expression ``gs.takeBy(x => x, 10)`` would return the array::

      [-1, 1, 2, 3, 4, 5, 6, 7, NA, NA]

    Returns the 10 samples with the least number of singletons:

    >>> samplesMostSingletons = (vds
    ...   .sample_qc()
    ...   .query_samples('samples.takeBy(s => sa.qc.nSingleton, 10)'))

    """

  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Int](f, n),
    integralTakeByDocs, "f" -> "Lambda expression for mapping an aggregable to an ordered value.", "n" -> "Number of items to take."
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedInt32Hr), int32Hr, arrayHr(TTHr))
  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Long](f, n),
    integralTakeByDocs, "f" -> "Lambda expression for mapping an aggregable to an ordered value.", "n" -> "Number of items to take."
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedInt64Hr), int32Hr, arrayHr(TTHr))

  private val floatingPointTakeByDocs = genericTakeByDocs ++ """
    Note that ``NaN`` always appears after any finite or infinite floating-point
    numbers but before ``NA``. For example, consider an aggregable containing
    these elements::

      Infinity, -1, 1, 0, -Infinity, NA, NaN

    The expression ``gs.takeBy(x => x, 7)`` would return the array::

      [-Infinity, -1, 0, 1, Infinity, NaN, NA]

    The expression ``gs.takeBy(x => -x, 7)`` would return the array::

      [Infinity, 1, 0, -1, -Infinity, NaN, NA]

    """

  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Float](f, n),
    floatingPointTakeByDocs, "f" -> "Lambda expression for mapping an aggregable to an ordered value.", "n" -> "Number of items to take."
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedFloat32Hr), int32Hr, arrayHr(TTHr))
  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Double](f, n),
    floatingPointTakeByDocs, "f" -> "Lambda expression for mapping an aggregable to an ordered value.", "n" -> "Number of items to take."
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedFloat64Hr), int32Hr, arrayHr(TTHr))

  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[String](f, n),
    genericTakeByDocs, "f" -> "Lambda expression for mapping an aggregable to an ordered value.", "n" -> "Number of items to take."
  )(aggregableHr(TTHr), unaryHr(TTHr, stringHr), int32Hr, arrayHr(TTHr))

  val aggST = Box[SymbolTable]()

  registerLambdaAggregatorTransformer("flatMap", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Unit) =>
    a { x =>
      val r = f(x).asInstanceOf[IndexedSeq[Any]]
      var i = 0
      while (i < r.size) {
        k(r(i))
        i += 1
      }
    }
  }
  }, { (x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) =>
    for (
      is <- f(x);
      (str, r) <- CM.memoize(Code.checkcast[IndexedSeq[AnyRef]](is));
      (stn, n) <- CM.memoize(r.invoke[Int]("size"));
      i <- CM.newLocal[Int];
      ri = r.invoke[Int, AnyRef]("apply", i);
      invokek <- k(ri)
    ) yield Code(
      str,
      stn,
      i.store(0),
      Code.whileLoop(i < n,
        Code(invokek, i.store(i + 1))
      )
    )
  }
  },
    """
    Returns a new aggregable by applying a function ``f`` to each element and concatenating the resulting arrays.

    **Examples**

    Compute a list of genes per sample with loss of function variants (result may have duplicate entries):

    >>> vds_result = vds.annotate_samples_expr('sa.lof_genes = gs.filter(g => va.consequence == "LOF" && g.nNonRefAlleles() > 0).flatMap(g => va.genes).collect()')
    """
  )(aggregableHr(TTHr, aggST), unaryHr(TTHr, arrayHr(TUHr)), aggregableHr(TUHr, aggST))

  registerLambdaAggregatorTransformer("flatMap", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Any) => a { x => f(x).asInstanceOf[Set[Any]].foreach(k) } }
  }, { (x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) =>
    for (
      fx <- f(x);
      (stit, it) <- CM.memoize(Code.checkcast[Set[AnyRef]](fx).invoke[Iterator[AnyRef]]("iterator"));
      hasNext = it.invoke[Boolean]("hasNext");
      next = it.invoke[AnyRef]("next");
      invokek <- k(next)
    ) yield Code(stit, Code.whileLoop(hasNext, invokek))
  }
  },
    """
    Returns a new aggregable by applying a function ``f`` to each element and concatenating the resulting sets.

    Compute a list of genes per sample with loss of function variants (result does not have duplicate entries):

    >>> vds_result = vds.annotate_samples_expr('sa.lof_genes = gs.filter(g => va.consequence == "LOF" && g.nNonRefAlleles() > 0).flatMap(g => va.genes.toSet()).collect()')
    """, "f" -> "Lambda expression."
  )(aggregableHr(TTHr, aggST), unaryHr(TTHr, setHr(TUHr)), aggregableHr(TUHr, aggST))

  registerLambdaAggregatorTransformer("filter", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Any) =>
    a { x =>
      val r = f(x)
      if (r != null && r.asInstanceOf[Boolean])
        k(x)
    }
  }
  }, { (_x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) =>
    for (
      (stx, x) <- CM.memoize(_x);
      (str, r) <- CM.memoize(f(x));
      invokek <- k(x)
    ) yield Code(stx, str,
      // NB: the invocation of `k` doesn't modify the stack.
      r.ifNull(Code._empty[Unit],
        Code.booleanValue(Code.checkcast[java.lang.Boolean](r)).mux(
          invokek,
          Code._empty[Unit]))
    )
  }
  },
    """
    Subsets an aggregable by evaluating ``f`` for each element and keeping those elements that evaluate to true.

    **Examples**

    Compute Hardy Weinberg Equilibrium for cases only:

    >>> vds_result = vds.annotate_variants_expr("va.hweCase = gs.filter(g => sa.isCase).hardyWeinberg()")
    """, "f" -> "Boolean lambda expression."
  )(aggregableHr(TTHr, aggST), unaryHr(TTHr, boolHr), aggregableHr(TTHr, aggST))

  registerLambdaAggregatorTransformer("map", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Any) => a { x => k(f(x)) } }
  }, { (x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) => f(x).flatMap(k) }
  },
    """
    Change the type of an aggregable by evaluating ``f`` for each element.

    **Examples**

    Convert an aggregable of genotypes (``gs``) to an aggregable of genotype quality scores and then compute summary statistics:

    >>> vds_result = vds.annotate_variants_expr("va.gqStats = gs.map(g => g.gq).stats()")
    """, "f" -> "Lambda expression."
  )(aggregableHr(TTHr, aggST), unaryHr(TTHr, TUHr), aggregableHr(TUHr, aggST))

  type Id[T] = T

  def registerNumericCode[T >: Null, S >: Null](name: String, f: (Code[T], Code[T]) => Code[S])(implicit hrt: HailRep[T], hrs: HailRep[S], tti: TypeInfo[T], sti: TypeInfo[S], tct: ClassTag[T], sct: ClassTag[S]) {
    val hrboxedt = new HailRep[T] {
      def typ: Type = hrt.typ
    }
    val hrboxeds = new HailRep[S] {
      def typ: Type = hrs.typ
    }

    registerCode(name, (x: Code[T], y: Code[T]) => CM.ret(f(x, y)), null)

    registerCode(name, (xs: Code[IndexedSeq[T]], y: Code[T]) =>
      CM.mapIS(xs, (xOpt: Code[T]) => xOpt.mapNull((x: Code[T]) => f(x, y))),
      null)(arrayHr(hrboxedt), hrt, arrayHr(hrboxeds))

    registerCode(name, (x: Code[T], ys: Code[IndexedSeq[T]]) =>
      CM.mapIS(ys, (yOpt: Code[T]) =>
        yOpt.mapNull((y: Code[T]) => f(x, y))),
      null)(hrt, arrayHr(hrboxedt), arrayHr(hrboxeds))

    registerCode(name, (xs: Code[IndexedSeq[T]], ys: Code[IndexedSeq[T]]) => for (
      (stn, n) <- CM.memoize(xs.invoke[Int]("size"));
      n2 = ys.invoke[Int]("size");

      (stb, b) <- CM.memoize(Code.newArray[S](n));

      i <- CM.newLocal[Int];

      (stx, x) <- CM.memoize(xs.invoke[Int, T]("apply", i));
      (sty, y) <- CM.memoize(ys.invoke[Int, T]("apply", i));

      z = x.mapNull(y.mapNull(f(x, y)))
    ) yield Code(stn,
      (n.ceq(n2)).mux(
        Code(
          i.store(0),
          stb,
          Code.whileLoop(i < n,
            Code(stx, sty, b.update(i, z), i.store(i + 1))
          ),
          CompilationHelp.arrayToWrappedArray(b)).asInstanceOf[Code[IndexedSeq[S]]],
        Code._throw(Code.newInstance[is.hail.utils.HailException, String, Option[String], Throwable](
          s"""Cannot apply operation $name to arrays of unequal length.""".stripMargin,
          Code.invokeStatic[scala.Option[String], scala.Option[String]]("empty"),
          Code._null[Throwable])))),
      null)
  }

  def registerNumeric[T, S](name: String, f: (T, T) => S)(implicit hrt: HailRep[T], hrs: HailRep[S]) {
    val hrboxedt = new HailRep[Any] {
      def typ: Type = hrt.typ
    }
    val hrboxeds = new HailRep[Any] {
      def typ: Type = hrs.typ
    }

    register(name, f, null)

    register(name, (x: IndexedSeq[Any], y: T) =>
      x.map { xi =>
        if (xi == null)
          null
        else
          f(xi.asInstanceOf[T], y)
      }, null)(arrayHr(hrboxedt), hrt, arrayHr(hrboxeds))

    register(name, (x: T, y: IndexedSeq[Any]) => y.map { yi =>
      if (yi == null)
        null
      else
        f(x, yi.asInstanceOf[T])
    }, null)(hrt, arrayHr(hrboxedt), arrayHr(hrboxeds))

    register(name, { (x: IndexedSeq[Any], y: IndexedSeq[Any]) =>
      if (x.length != y.length) fatal(
        s"""Cannot apply operation $name to arrays of unequal length:
           |  Left: ${ x.length } elements
           |  Right: ${ y.length } elements""".stripMargin)
      (x, y).zipped.map { case (xi, yi) =>
        if (xi == null || yi == null)
          null
        else
          f(xi.asInstanceOf[T], yi.asInstanceOf[T])
      }
    }, null)(arrayHr(hrboxedt), arrayHr(hrboxedt), arrayHr(hrboxeds))
  }

  registerMethod("toInt32", (s: String) => s.toInt, "Convert value to a 32-bit integer.")
  registerMethod("toInt64", (s: String) => s.toLong, "Convert value to a 64-bit integer.")
  registerMethod("toFloat32", (s: String) => s.toFloat, "Convert value to a 32-bit floating point number.")
  registerMethod("toFloat64", (s: String) => s.toDouble, "Convert value to a 64-bit floating point number.")

  registerMethod("toInt32", (b: Boolean) => b.toInt, "Convert value to a 32-bit integer. Returns 1 if true, else 0.")
  registerMethod("toInt64", (b: Boolean) => b.toLong, "Convert value to a 64-bit integer. Returns 1 if true, else 0.")
  registerMethod("toFloat32", (b: Boolean) => b.toFloat, "Convert value to a 32-bit floating point number. Returns 1.0 if true, else 0.0.")
  registerMethod("toFloat64", (b: Boolean) => b.toDouble, "Convert value to a 64-bit floating point number. Returns 1.0 if true, else 0.0.")

  def registerNumericType[T]()(implicit ev: Numeric[T], hrt: HailRep[T]) {
    // registerNumeric("+", ev.plus)
    registerNumeric("-", ev.minus)
    registerNumeric("*", ev.times)
    // registerNumeric("/", (x: T, y: T) => ev.toDouble(x) / ev.toDouble(y))

    registerMethod("abs", ev.abs _, "Returns the absolute value of a number.")
    registerMethod("signum", ev.signum _, "Returns the sign of a number (1, 0, or -1).")

    register("-", ev.negate _, "Returns the negation of this value.")
    register("fromInt", ev.fromInt _, null)

    registerMethod("toInt32", ev.toInt _, "Convert value to a 32-bit integer.")
    registerMethod("toInt64", ev.toLong _, "Convert value to a 64-bit integer.")
    registerMethod("toFloat32", ev.toFloat _, "Convert value to a 32-bit floating point number.")
    registerMethod("toFloat64", ev.toDouble _, "Convert value to a 64-bit floating point number.")
  }

  registerNumeric("**", (x: Double, y: Double) => math.pow(x, y))

  registerNumericCode("/", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => Code.boxDouble(Code.intValue(x).toD / Code.intValue(y).toD))
  registerNumericCode("/", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => Code.boxDouble(Code.longValue(x).toD / Code.longValue(y).toD))
  registerNumericCode("/", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => Code.boxDouble(Code.floatValue(x).toD / Code.floatValue(y).toD))
  registerNumericCode("/", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => Code.boxDouble(Code.doubleValue(x).toD / Code.doubleValue(y).toD))

  registerNumericCode("+", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => Code.boxInt(Code.intValue(x) + Code.intValue(y)))
  registerNumericCode("+", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => Code.boxLong(Code.longValue(x) + Code.longValue(y)))
  registerNumericCode("+", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => Code.boxFloat(Code.floatValue(x) + Code.floatValue(y)))
  registerNumericCode("+", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => Code.boxDouble(Code.doubleValue(x) + Code.doubleValue(y)))

  registerNumericType[Int]()
  registerNumericType[Long]()
  registerNumericType[Float]()
  registerNumericType[Double]()

  register("==", (a: Any, b: Any) => a == b, null)(TTHr, TUHr, boolHr)
  register("!=", (a: Any, b: Any) => a != b, null)(TTHr, TUHr, boolHr)

  def registerOrderedType[T]()(implicit ord: Ordering[T], hrt: HailRep[T]) {
    val hrboxedt = new HailRep[Any] {
      def typ: Type = hrt.typ
    }

    def extOrd(ascending: Boolean): Ordering[Any] = {
      val dirOrd = if (ascending) ord else ord.reverse
      extendOrderingToNull(missingGreatest = true)(
        new Ordering[Any] {
          def compare(a: Any, b: Any): Int = dirOrd.compare(a.asInstanceOf[T], b.asInstanceOf[T])
        })
    }

    // register("<", ord.lt _, null)
    register("<=", ord.lteq _, null)
    // register(">", ord.gt _, null)
    register(">=", ord.gteq _, null)

    registerMethod("min", ord.min _, "Returns the minimum value.")
    registerMethod("max", ord.max _, "Returns the maximum value.")

    registerMethod("sort", (a: IndexedSeq[Any]) => a.sorted(extOrd(ascending = true)), "Sort the collection in ascending order.")(arrayHr(hrboxedt), arrayHr(hrboxedt))
    registerMethod("sort", (a: IndexedSeq[Any], ascending: Boolean) =>
      a.sorted(extOrd(ascending)), "Sort the collection with the ordering specified by ``ascending``.", "ascending" -> "If true, sort the collection in ascending order. Otherwise, sort in descending order."
    )(arrayHr(hrboxedt), boolHr, arrayHr(hrboxedt))

    registerLambdaMethod("sortBy", (a: IndexedSeq[Any], f: (Any) => Any) =>
      a.sortBy(f)(extOrd(ascending = true)), "Sort the collection in ascending order after evaluating ``f`` for each element.", "f" -> "Lambda expression."
    )(arrayHr(TTHr), unaryHr(TTHr, hrboxedt), arrayHr(TTHr))

    registerLambdaMethod("sortBy", (a: IndexedSeq[Any], f: (Any) => Any, ascending: Boolean) =>
      a.sortBy(f)(extOrd(ascending)), "Sort the collection with the ordering specified by ``ascending`` after evaluating ``f`` for each element.",
      "f" -> "Lambda expression.", "ascending" -> "If true, sort the collection in ascending order. Otherwise, sort in descending order."
    )(arrayHr(TTHr), unaryHr(TTHr, hrboxedt), boolHr, arrayHr(TTHr))
  }

  register("<", implicitly[Ordering[Boolean]].lt _, null)
  registerCode("<", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => CM.ret(Code.boxBoolean(Code.intValue(x) < Code.intValue(y))), null)
  registerCode("<", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => CM.ret(Code.boxBoolean(Code.longValue(x) < Code.longValue(y))), null)
  registerCode("<", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => CM.ret(Code.boxBoolean(Code.floatValue(x) < Code.floatValue(y))), null)
  registerCode("<", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => CM.ret(Code.boxBoolean(Code.doubleValue(x) < Code.doubleValue(y))), null)
  register("<", implicitly[Ordering[String]].lt _, null)

  register(">", implicitly[Ordering[Boolean]].lt _, null)
  registerCode(">", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => CM.ret(Code.boxBoolean(Code.intValue(x) > Code.intValue(y))), null)
  registerCode(">", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => CM.ret(Code.boxBoolean(Code.longValue(x) > Code.longValue(y))), null)
  registerCode(">", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => CM.ret(Code.boxBoolean(Code.floatValue(x) > Code.floatValue(y))), null)
  registerCode(">", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => CM.ret(Code.boxBoolean(Code.doubleValue(x) > Code.doubleValue(y))), null)
  register(">", implicitly[Ordering[String]].lt _, null)

  registerOrderedType[Boolean]()
  registerOrderedType[Int]()
  registerOrderedType[Long]()
  registerOrderedType[Float]()
  registerOrderedType[Double]()
  registerOrderedType[String]()

  register("//", (x: Int, y: Int) => java.lang.Math.floorDiv(x, y), null)
  register("//", (x: Long, y: Long) => java.lang.Math.floorDiv(x, y), null)
  register("//", (x: Float, y: Float) => math.floor(x / y).toFloat, null)
  register("//", (x: Double, y: Double) => math.floor(x / y), null)

  register("%", (x: Int, y: Int) => java.lang.Math.floorMod(x, y), null)
  register("%", (x: Long, y: Long) => java.lang.Math.floorMod(x, y), null)
  register("%", (x: Float, y: Float) => {
    val t = x % y
    if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0) t else t + y
  }, null)
  register("%", (x: Double, y: Double) => {
    val t = x % y
    if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0) t else t + y
  }, null)

  register("+", (x: String, y: Any) => x + y, null)(stringHr, TTHr, stringHr)

  register("~", (s: String, t: String) => s.r.findFirstIn(t).isDefined, null)

  register("isnan", (d: Double) => d.isNaN, "Returns true if the argument is NaN (not a number), false if the argument is defined but not NaN. Returns missing if the argument is missing.")

  registerSpecial("isMissing", (g: () => Any) => g() == null, "Returns true if item is missing. Otherwise, false.")(TTHr, boolHr)
  registerSpecial("isDefined", (g: () => Any) => g() != null, "Returns true if item is non-missing. Otherwise, false.")(TTHr, boolHr)

  private def nullableBooleanCode(isOr: Boolean)
    (left: Code[java.lang.Boolean], right: Code[java.lang.Boolean]): Code[java.lang.Boolean] = {
    val (isZero, isNotZero) =
      if (isOr)
        (IFNE, IFEQ)
      else
        (IFEQ, IFNE)

    new Code[java.lang.Boolean] {
      // AND:
      // true true => true
      // true false => false
      // true null => null

      // null true => null
      // null false => false
      // null null => null

      // false true => false
      // false false => false
      // false null => false

      // OR:
      // true true => true
      // true false => true
      // true null => true

      // null true => true
      // null false => null
      // null null => null

      // false true => true
      // false false => false
      // false null => null
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        val lnullorfalse = new LabelNode
        val ldone = new LabelNode
        val lfirst = new LabelNode
        val lsecond = new LabelNode

        left.emit(il) // L
        il += new InsnNode(DUP) // L L
        il += new JumpInsnNode(IFNULL, lnullorfalse) // L
        il += new InsnNode(DUP) // L L
        (Code._empty[java.lang.Boolean].invoke[Boolean]("booleanValue")).emit(il) // L Z
        il += new JumpInsnNode(isZero, ldone) // L

        // left = null or false
        il += lnullorfalse // L
        il += new InsnNode(DUP) // L L
        right.emit(il) // L L R
        il += new InsnNode(SWAP) // L R L
        il += new JumpInsnNode(IFNONNULL, lfirst) // L R; stack indexing is from right to left

        // left = null
        il += new InsnNode(DUP) // L R R
        il += new JumpInsnNode(IFNULL, lsecond) // L R; both are null so either one works
        il += new InsnNode(DUP) // L R R
        (Code._empty[java.lang.Boolean].invoke[Boolean]("booleanValue")).emit(il) // L R Z
        il += new JumpInsnNode(isNotZero, lsecond) // L R; stack indexing is from right to left

        il += lfirst // B A
        il += new InsnNode(SWAP) // A B
        il += lsecond // A B
        il += new InsnNode(POP) // A
        il += ldone // A
      }
    }
  }

  registerSpecialCode("||", (a: Code[java.lang.Boolean], b: Code[java.lang.Boolean]) =>
    CM.ret(nullableBooleanCode(true)(a, b)), null)
  registerSpecialCode("&&", (a: Code[java.lang.Boolean], b: Code[java.lang.Boolean]) =>
    CM.ret(nullableBooleanCode(false)(a, b)), null)

  registerSpecial("orElse", { (f1: () => Any, f2: () => Any) =>
    val v = f1()
    if (v == null)
      f2()
    else
      v
  },
    """
    If ``a`` is not missing, returns ``a``. Otherwise, returns ``b``.

    **Examples**

    Replace missing phenotype values with the mean value:

    ::

        >>> [mean_height] = vds.query_samples(['samples.map(s => sa.pheno.height).stats()'])['mean']
        >>> vds.annotate_samples_expr('sa.pheno.heightImputed = orElse(sa.pheno.height, %d)' % mean_height)
    """
  )(TTHr, TTHr, TTHr)

  register("orMissing", { (predicate: Boolean, value: Any) =>
    if (predicate)
      value
    else
      null
  },
    """
    If ``predicate`` evaluates to true, returns ``value``. Otherwise, returns NA.
    """
  )(boolHr, TTHr, TTHr)

  registerMethodCode("[]", (a: Code[IndexedSeq[AnyRef]], i: Code[java.lang.Integer]) => for (
    (storei, refi) <- CM.memoize(Code.intValue(i));
    size = a.invoke[Int]("size")
  ) yield {
    Code(storei, a.invoke[Int, AnyRef]("apply", (refi >= 0).mux(refi, refi + size)))
  },
    """
    Returns the i*th* element (0-indexed) of the array, or throws an exception if ``i`` is an invalid index.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[2]
        result: 4
    """, "i" -> "Index of the element to return."
  )(arrayHr(BoxedTTHr), boxedInt32Hr, BoxedTTHr)
  registerMethod("[]", (a: Map[Any, Any], i: Any) => a(i),
    """
    Returns the value for ``k``, or throws an exception if the key is not found.
    """, "k" -> "Key in the Dict to query."
  )(dictHr(TTHr, TUHr), TTHr, TUHr)
  registerMethod("[]", (a: String, i: Int) => (if (i >= 0) a(i) else a(a.length + i)).toString,
    """
    Returns the i*th* element (0-indexed) of the string, or throws an exception if ``i`` is an invalid index.

    .. code-block:: text
        :emphasize-lines: 2

        let s = "genetics" in s[6]
        result: "c"
    """, "i" -> "Index of the character to return."
  )(stringHr, int32Hr, stringHr)

  registerMethod("[:]", (a: IndexedSeq[Any]) => a,
    """
    Returns a copy of the array.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6] in a[:]
        result: [0, 2, 4, 6]
    """
  )(arrayHr(TTHr), arrayHr(TTHr))
  registerMethod("[*:]", (a: IndexedSeq[Any], i: Int) => a.slice(if (i < 0) a.length + i else i, a.length),
    """
    Returns a slice of the array from the i*th* element (0-indexed) to the
    end. Negative indices are interpreted as offsets from the end of the array.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[3:]
        result: [6, 8, 10]

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[-5:]
        result: [2, 4, 6, 8, 10]
    """, "i" -> "Starting index of the slice."
  )(arrayHr(TTHr), int32Hr, arrayHr(TTHr))
  registerMethod("[:*]", (a: IndexedSeq[Any], i: Int) => a.slice(0, if (i < 0) a.length + i else i),
    """

    Returns a slice of the array from the first element until the j*th* element
    (0-indexed). Negative indices are interpreted as offsets from the end
    of the array.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[:4]
        result: [0, 2, 4, 6]

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[:-4]
        result: [0, 2]
    """, "j" -> "End index of the slice (not included in result)."
  )(arrayHr(TTHr), int32Hr, arrayHr(TTHr))
  registerMethod("[*:*]", (a: IndexedSeq[Any], i: Int, j: Int) => a.slice(if (i < 0) a.length + i else i, if (j < 0) a.length + j else j),
    """
    Returns a slice of the array from the i*th* element until the j*th* element
    (both 0-indexed). Negative indices are interpreted as offsets from the end
    of the array.

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[2:4]
        result: [4, 6]

    .. code-block:: text
        :emphasize-lines: 2

        let a = [0, 2, 4, 6, 8, 10] in a[-3:-1]
        result: [6, 8]

    A handy way to understand the behavior of negative indicies is to recall
    this rule:

    .. code-block:: text

        s[-i:-j] == s[s.length - i, s.length - j]
    """, "i" -> "Starting index of the slice.", "j" -> "End index of the slice (not included in result)."
  )(arrayHr(TTHr), int32Hr, int32Hr, arrayHr(TTHr))

  registerMethod("[:]", (a: String) => a,
    """
    Returns a copy of the string.

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcd" in a[:]
        result: "abcd"
    """
  )

  registerMethod("[*:]", (a: String, i: Int) => a.slice(if (i < 0) a.length + i else i, a.length),
    """
    Returns a slice of the string from the i*th* unicode-codepoint (0-indexed)
    to the end. Negative indices are interpreted as offsets from the end of the
    string.

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcdef" in a[3:]
        result: "def"

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcdef" in a[-5:]
        result: "bcdef"
    """, "i" -> "Starting index of the slice."
  )

  registerMethod("[:*]", (a: String, i: Int) => a.slice(0, if (i < 0) a.length + i else i),
    """
    Returns a slice of the string from the first unicode-codepoint until the
    j*th* unicode-codepoint (0-indexed). Negative indices are interpreted as
    offsets from the end of the string.

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcdef" in a[:4]
        result: "abcd"

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcdef" in a[:-3]
        result: "abc"
    """, "j" -> "End index of the slice (not included in result)."
  )

  registerMethod("[*:*]", (a: String, i: Int, j: Int) => a.slice(if (i < 0) a.length + i else i, if (j < 0) a.length + j else j),
    """
    Returns a slice of the string from the i*th* unicode-codepoint until the
    j*th* unicode-codepoint (both 0-indexed). Negative indices are interpreted
    as offsets from the end of the string instead of the beginning.

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcdef" in a[2:4]
        result: "cd"

    .. code-block:: text
        :emphasize-lines: 2

        let a = "abcdef" in a[-3:-1]
        result: "de"

    A handy way to understand the behavior of negative indicies is to recall
    this rule, which holds for any positive integers i and j:

    .. code-block:: text

        s[-i:-j] == s[s.length - i, s.length - j]
    """, "i" -> "Starting index of the slice.", "j" -> "End index of the slice (not included in result)."
  )
}
