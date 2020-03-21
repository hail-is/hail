package is.hail.asm4s

import java.io._

import is.hail.utils._
import is.hail.lir

import org.apache.spark.TaskContext
import org.objectweb.asm.tree._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.util.{Textifier, TraceClassVisitor}
import org.objectweb.asm.ClassReader

import scala.collection.mutable
import scala.reflect.ClassTag

class Field[T: TypeInfo](classBuilder: ClassBuilder[_], val name: String) {
  val lf: lir.Field = classBuilder.lclass.newField(name, typeInfo[T])

  def get(obj: Code[_]): Code[T] = Code(obj, lir.getField(lf))

  def put(obj: Code[_], v: Code[T]): Code[Unit] = {
    obj.end.append(lir.goto(v.start))
    v.end.append(lir.putField(lf, obj.v, v.v))
    val newC = new VCode(obj.start, v.end, null)
    obj.clear()
    v.clear()
    newC
  }
}

class ClassesBytes(classesBytes: Array[(String, Array[Byte])]) extends Serializable {
  @transient @volatile var loaded: Boolean = false

  def load(): Unit = {
    if (!loaded) {
      synchronized {
        if (!loaded) {
          classesBytes.foreach { case (n, bytes) =>
            try {
              HailClassLoader.loadOrDefineClass(n, bytes)
            } catch {
              case e: Exception =>
                FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
                throw e
            }
          }
        }
        loaded = true
      }
    }
  }
}

trait WrappedModuleBuilder {
  def modb: ModuleBuilder

  def newClass[C](name: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = modb.newClass[C](name)

  def genClass[C](baseName: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = modb.genClass[C](baseName)

  def classesBytes(print: Option[PrintWriter] = None): ClassesBytes = modb.classesBytes(print)
}

class ModuleBuilder() {
  val classes = new mutable.ArrayBuffer[ClassBuilder[_]]()

  def newClass[C](name: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = {
    val c = new ClassBuilder[C](this, name)
    c.addInterface(cti.iname)
    classes += c
    c
  }

  def genClass[C](baseName: String)(implicit cti: TypeInfo[C]): ClassBuilder[C] = newClass[C](genName("C", baseName))

  var classesBytes: ClassesBytes = _

  def classesBytes(print: Option[PrintWriter] = None): ClassesBytes = {
    if (classesBytes == null) {
      classesBytes = new ClassesBytes(
        classes
          .iterator
          .map(c => (c.name.replace("/", "."), c.classAsBytes(print)))
          .toArray)

    }
    classesBytes
  }
}

trait WrappedClassBuilder[C] extends WrappedModuleBuilder {
  def cb: ClassBuilder[C]

  def modb: ModuleBuilder = cb.modb

  def name: String = cb.name

  def ti: TypeInfo[_] = cb.ti

  def addInterface(name: String): Unit = cb.addInterface(name)

  def emitInit(c: Code[Unit]): Unit = cb.emitInit(c)

  def newField[T: TypeInfo](name: String): Field[T] = cb.newField[T](name)

  def genField[T: TypeInfo](baseName: String): Field[T] = cb.genField(baseName)

  def genFieldThisRef[T: TypeInfo](name: String = null): ThisFieldRef[T] = cb.genFieldThisRef[T](name)

  def genLazyFieldThisRef[T: TypeInfo](setup: Code[T], name: String = null): Value[T] = cb.genLazyFieldThisRef(setup, name)

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): Value[T] = cb.getOrDefineLazyField(setup, id)

  def newMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] =
    cb.newMethod(name, parameterTypeInfo, returnTypeInfo)

  def newMethod(name: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_]): MethodBuilder[C] =
    cb.newMethod(name, maybeGenericParameterTypeInfo, maybeGenericReturnTypeInfo)

  def getOrGenMethod(
    baseName: String, key: Any, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]
  )(body: MethodBuilder[C] => Unit): MethodBuilder[C] =
    cb.getOrGenMethod(baseName, key, argsInfo, returnInfo)(body)

  def result(print: Option[PrintWriter] = None): () => C = cb.result(print)

  def genMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder[C] =
    cb.genMethod(baseName, argsInfo, returnInfo)

  def genMethod[R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[R](baseName)

  def genMethod[A: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, A3, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, A3, A4, R](baseName)

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] = cb.genMethod[A1, A2, A3, A4, A5, R](baseName)
}

class ClassBuilder[C](
  val modb: ModuleBuilder,
  val name: String = null) extends WrappedModuleBuilder {

  val ti: TypeInfo[C] = new ClassInfo[C](name)

  val lclass = new lir.Classx[C](name, "java/lang/Object")

  val methods: mutable.ArrayBuffer[MethodBuilder[C]] = new mutable.ArrayBuffer[MethodBuilder[C]](16)
  val fields: mutable.ArrayBuffer[FieldNode] = new mutable.ArrayBuffer[FieldNode](16)

  val lazyFieldMemo: mutable.Map[Any, Value[_]] = mutable.Map.empty

  val lInit = lclass.newMethod("<init>", FastIndexedSeq(), UnitInfo)

  var initBody: Code[Unit] = {
    val L = new lir.Block()
    L.append(
      lir.methodStmt(INVOKESPECIAL,
        "java/lang/Object",
        "<init>",
        "()V",
        false,
        UnitInfo,
        FastIndexedSeq(lir.load(lInit.getParam(0)))))
    L.append(lir.returnx())
    new VCode(L, L, null)
  }

  def emitInit(c: Code[Unit]): Unit = {
    initBody = Code(initBody, c)
  }

  def addInterface(name: String): Unit = lclass.addInterface(name)

  def newMethod(name: String, parameterTypeInfo: IndexedSeq[TypeInfo[_]], returnTypeInfo: TypeInfo[_]): MethodBuilder[C] = {
    val mb = new MethodBuilder[C](this, name, parameterTypeInfo, returnTypeInfo)
    methods.append(mb)
    mb
  }

  def newMethod(name: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_]): MethodBuilder[C] = {

    val parameterTypeInfo: IndexedSeq[TypeInfo[_]] = maybeGenericParameterTypeInfo.map(_.base)
    val returnTypeInfo: TypeInfo[_] = maybeGenericReturnTypeInfo.base
    val m = newMethod(name, parameterTypeInfo, returnTypeInfo)
    if (maybeGenericParameterTypeInfo.exists(_.isGeneric) || maybeGenericReturnTypeInfo.isGeneric) {
      val generic = newMethod(name, maybeGenericParameterTypeInfo.map(_.generic), maybeGenericReturnTypeInfo.generic)
      generic.emit(
        maybeGenericReturnTypeInfo.castToGeneric(
          m.invoke(maybeGenericParameterTypeInfo.zipWithIndex.map { case (ti, i) =>
            ti.castFromGeneric(generic.getArg(i + 1)(ti.generic))
          }: _*)))
    }
    m
  }

  def genDependentFunction[A1 : TypeInfo, R : TypeInfo](baseName: String): DependentFunction[AsmFunction1[A1, R]] = {
    val depCB = modb.genClass[AsmFunction1[A1, R]](baseName)
    val apply = depCB.newMethod("apply", Array(GenericTypeInfo[A1]), GenericTypeInfo[R])
    new DependentFunctionBuilder[AsmFunction1[A1, R]](apply)
  }

  def newField[T: TypeInfo](name: String): Field[T] = new Field[T](this, name)

  def genField[T: TypeInfo](baseName: String): Field[T] = newField(genName("f", baseName))

  private[this] val methodMemo: mutable.Map[Any, MethodBuilder[C]] = mutable.HashMap.empty

  def getOrGenMethod(baseName: String, key: Any, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_])
    (f: MethodBuilder[C] => Unit): MethodBuilder[C] = {
    methodMemo.get(key) match {
      case Some(mb) => mb
      case None =>
        val mb = newMethod(genName("M", baseName), argsInfo, returnInfo)
        f(mb)
        methodMemo(key) = mb
        mb
    }
  }

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = {
    assert(initBody.start != null)
    lInit.setEntry(initBody.start)

    lclass.asBytes(print)
  }

  def result(print: Option[PrintWriter] = None): () => C = {
    val n = name.replace("/", ".")
    val classesBytes = modb.classesBytes()

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    new (() => C) with java.io.Serializable {
      @transient @volatile private var theClass: Class[_] = null

      def apply(): C = {
        if (theClass == null) {
          this.synchronized {
            if (theClass == null) {
              classesBytes.load()
              theClass = loadClass(n)
            }
          }
        }

        theClass.newInstance().asInstanceOf[C]
      }
    }
  }

  def _this: Value[C] = new LocalRef[C](new lir.Parameter(null, 0, ti))(ti)

  def genFieldThisRef[T: TypeInfo](name: String = null): ThisFieldRef[T] =
    new ThisFieldRef[T](this, genField[T](name))

  def genLazyFieldThisRef[T: TypeInfo](setup: Code[T], name: String = null): Value[T] =
    new ThisLazyFieldRef[T](this, name, setup)

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): Value[T] = {
    lazyFieldMemo.getOrElseUpdate(id, genLazyFieldThisRef[T](setup)).asInstanceOf[ThisLazyFieldRef[T]]
  }
  
  def genMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder[C] =
    newMethod(genName("m", baseName), argsInfo, returnInfo)
  
  def genMethod[R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](), typeInfo[R])

  def genMethod[A: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2]), typeInfo[R])
  
  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4]), typeInfo[R])

  def genMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): MethodBuilder[C] =
    genMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4], typeInfo[A5]), typeInfo[R])
}

object FunctionBuilder {
  val stderrAndLoggerErrorOS = getStderrAndLogOutputStream[FunctionBuilder[_]]

  def bytesToBytecodeString(bytes: Array[Byte], out: OutputStream) {
    val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(out))
    new ClassReader(bytes).accept(tcv, 0)
  }

  def apply[F](
    baseName: String,
    argInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    returnInfo: MaybeGenericTypeInfo[_]
  )(implicit fti: TypeInfo[F]): FunctionBuilder[F] = {
    val modb: ModuleBuilder = new ModuleBuilder()
    val cb: ClassBuilder[F] = modb.newClass[F](genName("C", baseName))
    val apply = cb.newMethod("apply", argInfo, returnInfo)
    new FunctionBuilder[F](apply)
  }

  def apply[R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction0[R]] =
    apply[AsmFunction0[R]](baseName, FastIndexedSeq.empty[MaybeGenericTypeInfo[_]], GenericTypeInfo[R])

  def apply[A1: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction1[A1, R]] =
    apply[AsmFunction1[A1, R]](baseName, Array(GenericTypeInfo[A1]), GenericTypeInfo[R])

  def apply[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction2[A1, A2, R]] =
    apply[AsmFunction2[A1, A2, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])

  def apply[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction3[A1, A2, A3, R]] =
    apply[AsmFunction3[A1, A2, A3, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])

  def apply[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): FunctionBuilder[AsmFunction4[A1, A2, A3, A4, R]] =
    apply[AsmFunction4[A1, A2, A3, A4, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3], GenericTypeInfo[A4]), GenericTypeInfo[R])
}

trait WrappedMethodBuilder[C] extends WrappedClassBuilder[C] {
  def mb: MethodBuilder[C]

  def cb: ClassBuilder[C] = mb.cb

  def parameterTypeInfo: IndexedSeq[TypeInfo[_]] = mb.parameterTypeInfo

  def returnTypeInfo: TypeInfo[_] = mb.returnTypeInfo

  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] = mb.newLocal(name)

  def getArg[T: TypeInfo](i: Int): LocalRef[T] = mb.getArg[T](i)

  def emitStartup(c: Code[Unit]): Unit = mb.emitStartup(c)

  def emit(body: Code[_]): Unit = mb.emit(body)

  def invoke[T](args: Code[_]*): Code[T] = mb.invoke(args: _*)
}

class MethodBuilder[C](
  val cb: ClassBuilder[C], _mname: String,
  val parameterTypeInfo: IndexedSeq[TypeInfo[_]],
  val returnTypeInfo: TypeInfo[_]
) extends WrappedClassBuilder[C] {
  val mname: String = _mname.substring(0, scala.math.min(_mname.length, 65535))
  if (!isJavaIdentifier(mname))
    throw new IllegalArgumentException(s"Illegal method name, not Java identifier: $mname")

  val lmethod: lir.Method = cb.lclass.newMethod(mname, parameterTypeInfo, returnTypeInfo)

  def newLocal[T: TypeInfo](name: String = null): LocalRef[T] =
    new LocalRef[T](lmethod.newLocal(name, typeInfo[T]))

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef(lmethod.getParam(i))

  private var emitted = false

  private var startup: Code[Unit] = Code._empty

  def emitStartup(c: Code[Unit]): Unit = {
    assert(!emitted)
    startup = Code(startup, c)
  }

  def emit(body: Code[_]): Unit = {
    assert(!emitted)
    emitted = true

    val start = startup.start
    startup.end.append(lir.goto(body.start))
    body.end.append(
      if (body.v != null)
        lir.returnx(body.v)
      else
        lir.returnx())
    assert(start != null)
    lmethod.setEntry(start)

    body.clear()
  }

  def invoke[T](args: Code[_]*): Code[T] = {
    val (start, end, argvs) = Code.sequenceValues(args.toFastIndexedSeq)
    if (returnTypeInfo eq UnitInfo) {
      end.append(
        lir.methodStmt(INVOKEVIRTUAL, lmethod,
          lir.load(new lir.Parameter(null, 0, cb.ti)) +: argvs))
      new VCode(start, end, null)
    } else {
      new VCode(start, end,
        lir.methodInsn(INVOKEVIRTUAL, lmethod,
          lir.load(new lir.Parameter(null, 0, cb.ti)) +: argvs))
    }
  }
}

trait DependentFunction[F] extends WrappedMethodBuilder[F] {
  def mb: MethodBuilder[F]

  var setFields: mutable.ArrayBuffer[(lir.ValueX) => Code[Unit]] = new mutable.ArrayBuffer()

  def newDepField[T : TypeInfo](value: Code[T]): Value[T] = {
    val cfr = genFieldThisRef[T]()
    setFields += { (obj: lir.ValueX) =>
      value.end.append(lir.putField(cb.name, cfr.name, typeInfo[T], obj, value.v))
      val newC = new VCode(value.start, value.end, null)
      value.clear()
      newC
    }
    cfr
  }

  def newDepFieldAny[T](value: Code[_])(implicit ti: TypeInfo[T]): Value[T] =
    newDepField(value.asInstanceOf[Code[T]])

  def newInstance(mb: MethodBuilder[_]): Code[F] = {
    val L = new lir.Block()

    val obj = new lir.Local(null, "new_dep_fun", cb.ti)
    L.append(lir.store(obj, lir.newInstance(cb.ti)))
    L.append(lir.methodStmt(INVOKESPECIAL,
      cb.name, "<init>", "()V", false,
      UnitInfo,
      Array(lir.load(obj))))

    var end = L
    setFields.foreach { f =>
      val c = f(lir.load(obj))
      end.append(lir.goto(c.start))
      end = c.end
    }
    new VCode(L, end, lir.load(obj))
  }

  override def result(pw: Option[PrintWriter]): () => F =
    throw new UnsupportedOperationException("cannot call result() on a dependent function")
}

class DependentFunctionBuilder[F](apply_method: MethodBuilder[F]) extends WrappedMethodBuilder[F] with DependentFunction[F] {
  def mb: MethodBuilder[F] = apply_method
}

class FunctionBuilder[F](
  val apply_method: MethodBuilder[F]
) extends WrappedMethodBuilder[F] {
  val mb: MethodBuilder[F] = apply_method
}
