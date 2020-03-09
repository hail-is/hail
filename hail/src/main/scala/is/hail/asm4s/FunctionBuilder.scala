package is.hail.asm4s

import java.io._
import java.util

import is.hail.utils._
import org.apache.spark.TaskContext
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._
import org.objectweb.asm.util.{CheckClassAdapter, Textifier, TraceClassVisitor}
import org.objectweb.asm.{ClassReader, ClassWriter, Type}

import scala.collection.JavaConverters._
import scala.collection.generic.Growable
import scala.collection.mutable
import scala.reflect.ClassTag

class Field[T: TypeInfo](classBuilder: ClassBuilder[_], val name: String) {
  val desc: String = typeInfo[T].name
  val node: FieldNode = new FieldNode(ACC_PUBLIC, name, desc, null, null)
  classBuilder.addField(node)

  def get(obj: Code[_]): Code[T] =
    Code(obj, new FieldInsnNode(GETFIELD, classBuilder.name, name, desc))

  def put(obj: Code[_], v: Code[T]): Code[Unit] =
    Code(obj, v, new FieldInsnNode(PUTFIELD, classBuilder.name, name, desc))
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

class ModuleBuilder {
  val classes = new mutable.ArrayBuffer[ClassBuilder[_]]()

  def newClass[C](name: String = null): ClassBuilder[C] = {
    val c = new ClassBuilder[C](this, name)
    classes += c
    c
  }

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

class ClassBuilder[C](
  module: ModuleBuilder,
  val name: String = null) {
  // FIXME use newClass
  module.classes += this

  var nameCounter: Int = 0

  val cn = new ClassNode()

  val methods: mutable.ArrayBuffer[MethodBuilder] = new mutable.ArrayBuffer[MethodBuilder](16)
  val fields: mutable.ArrayBuffer[FieldNode] = new mutable.ArrayBuffer[FieldNode](16)

  val init = new MethodNode(ACC_PUBLIC, "<init>", "()V", null, null)

  val lazyFieldMemo: mutable.Map[Any, LazyFieldRef[_]] = mutable.Map.empty

  // init
  cn.version = V1_8
  cn.access = ACC_PUBLIC

  cn.name = name
  cn.superName = "java/lang/Object"
  cn.interfaces.asInstanceOf[java.util.List[String]].add("java/io/Serializable")

  cn.methods.asInstanceOf[util.List[MethodNode]].add(init)

  init.instructions.add(new IntInsnNode(ALOAD, 0))
  init.instructions.add(new MethodInsnNode(INVOKESPECIAL, Type.getInternalName(classOf[java.lang.Object]), "<init>", "()V", false))

  // methods
  def genName(tag: String): String = {
    nameCounter += 1
    s"__$tag$nameCounter"
  }

  def genName(tag: String, suffix: String): String = {
    if (suffix == null)
      return genName(tag)

    nameCounter += 1
    s"__$tag$nameCounter$suffix"
  }

  def addInitInstructions(c: Code[Unit]): Unit = {
    val l = new mutable.ArrayBuffer[AbstractInsnNode]()
    c.emit(l)
    l.foreach(init.instructions.add _)
  }

  def addInterface(name: String): Unit = {
    cn.interfaces.asInstanceOf[java.util.List[String]].add(name)
  }

  def addMethod(m: MethodBuilder): Unit = {
    methods.append(m)
  }

  def addField(node: FieldNode): Unit = {
    cn.fields.asInstanceOf[util.List[FieldNode]].add(node)
  }

  def newDependentFunction[A1 : TypeInfo, R : TypeInfo]: DependentFunction[AsmFunction1[A1, R]] = {
    new DependentFunctionBuilder[AsmFunction1[A1, R]](
      Array(GenericTypeInfo[A1]),
      GenericTypeInfo[R],
      initModule = module)
  }

  def newField[T: TypeInfo](name: String): Field[T] = new Field[T](this, name)

  def genField[T: TypeInfo](): Field[T] = newField[T](genName("f"))

  def genField[T: TypeInfo](suffix: String): Field[T] = newField(genName("f", suffix))

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = {
    init.instructions.add(new InsnNode(RETURN))

    val cw = new ClassWriter(ClassWriter.COMPUTE_MAXS + ClassWriter.COMPUTE_FRAMES)
    val sw1 = new StringWriter()
    var bytes: Array[Byte] = new Array[Byte](0)
    try {
      for (method <- cn.methods.asInstanceOf[util.List[MethodNode]].asScala) {
        val count = method.instructions.size
        log.info(s"instruction count: $count: ${ cn.name }.${ method.name }")
        if (count > 8000)
          log.warn(s"big method: $count: ${ cn.name }.${ method.name }")
      }

      cn.accept(cw)
      bytes = cw.toByteArray
      //       This next line should always be commented out!
      //      CheckClassAdapter.verify(new ClassReader(bytes), false, new PrintWriter(sw1))
    } catch {
      case e: Exception =>
        // if we fail with frames, try without frames for better error message
        val cwNoFrames = new ClassWriter(ClassWriter.COMPUTE_MAXS)
        val sw2 = new StringWriter()
        cn.accept(cwNoFrames)
        try {
          CheckClassAdapter.verify(new ClassReader(cwNoFrames.toByteArray), false, new PrintWriter(sw2))
        } catch {
          case e: Exception =>
            log.error("Verify Output 1 for " + name + ":")
            throw e
        }

        if (sw2.toString.length() != 0) {
          System.err.println("Verify Output 2 for " + name + ":")
          System.err.println(sw2)
          throw new IllegalStateException("Bytecode failed verification 1", e)
        } else {
          if (sw1.toString.length() != 0) {
            System.err.println("Verify Output 1 for " + name + ":")
            System.err.println(sw1)
          }
          throw e
        }
    }

    if (sw1.toString.length != 0) {
      System.err.println("Verify Output 1 for " + name + ":")
      System.err.println(sw1)
      throw new IllegalStateException("Bytecode failed verification 2")
    }

    print.foreach { pw =>
      val cr = new ClassReader(bytes)
      val tcv = new TraceClassVisitor(null, new Textifier, pw)
      cr.accept(tcv, 0)
    }
    bytes
  }

  def result(print: Option[PrintWriter] = None): () => C = {
    val n = name.replace("/", ".")
    val classesBytes = module.classesBytes()

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
}

object FunctionBuilder {
  val stderrAndLoggerErrorOS = getStderrAndLogOutputStream[FunctionBuilder[_]]

  var count = 0

  def newUniqueID(): Int = {
    val id = count
    count += 1
    id
  }

  def bytesToBytecodeString(bytes: Array[Byte], out: OutputStream) {
    val tcv = new TraceClassVisitor(null, new Textifier, new PrintWriter(out))
    new ClassReader(bytes).accept(tcv, 0)
  }

  def functionBuilder[R: TypeInfo]: FunctionBuilder[AsmFunction0[R]] =
    new FunctionBuilder(Array[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction1[A, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction2[A, B, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction3[A, B, C, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def functionBuilder[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo]: FunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    new FunctionBuilder(Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
}

object MethodBuilder {
  private val localInsnRefs = mutable.Map[AbstractInsnNode, LocalRef[_]]()

  def registerLocalInsn(x: AbstractInsnNode, lr: LocalRef[_]): Unit = {
    assert(!localInsnRefs.contains(x))
    localInsnRefs += (x -> lr)
  }

  def popInsnRef(x: AbstractInsnNode): Option[LocalRef[_]] = {
    val lr = localInsnRefs.get(x)
    lr match {
      case Some(_) =>
        localInsnRefs -= x
      case None =>
    }
    lr
  }
}

class MethodBuilder(val fb: FunctionBuilder[_], _mname: String, val parameterTypeInfo: IndexedSeq[TypeInfo[_]], val returnTypeInfo: TypeInfo[_]) {
  def descriptor: String = s"(${ parameterTypeInfo.map(_.name).mkString })${ returnTypeInfo.name }"

  val mname = {
    val s = _mname.substring(0, scala.math.min(_mname.length, 65535))
    require(java.lang.Character.isJavaIdentifierStart(s.head), "invalid java identifier, " + s)
    require(s.forall(java.lang.Character.isJavaIdentifierPart(_)), "invalid java identifer, " + s)
    s
  }

  val mn = new MethodNode(ACC_PUBLIC, mname, descriptor, null, null)
  fb.classBuilder.cn.methods.asInstanceOf[util.List[MethodNode]].add(mn)

  val start = new LabelNode
  val end = new LabelNode
  val layout: IndexedSeq[Int] = 0 +: (parameterTypeInfo.scanLeft(1) { case (prev, gti) => prev + gti.slots })
  val argIndex: IndexedSeq[Int] = layout.init
  var locals: Int = layout.last

  def allocateLocal(name: String)(implicit tti: TypeInfo[_]): Int = {
    val i = locals
    assert(i < (1 << 16))
    locals += tti.slots

    mn.localVariables.asInstanceOf[util.List[LocalVariableNode]]
      .add(new LocalVariableNode(if (name == null) "local" + i else name, tti.name, null, start, end, i))
    i
  }

  def newLocal[T](implicit tti: TypeInfo[T]): LocalRef[T] =
    newLocal()

  def newLocal[T](name: String = null)(implicit tti: TypeInfo[T]): LocalRef[T] =
    new LocalRef[T](this, name)

  def newField[T: TypeInfo]: ClassFieldRef[T] = newField[T]()

  def newField[T: TypeInfo](name: String = null): ClassFieldRef[T] = fb.newField[T](name)

  def newLazyField[T: TypeInfo](setup: Code[T], name: String = null): LazyFieldRef[T] = fb.newLazyField(setup, name)

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): Settable[T] = {
    assert(i >= 0)
    assert(i < layout.length)
    new ArgRef[T](argIndex(i))
  }

  private var emitted = false

  private val startup = new mutable.ArrayBuffer[AbstractInsnNode]()

  def emitStartup(c: Code[_]): Unit = {
    assert(!emitted)
    c.emit(startup)
  }

  def emit(c: Code[_]) {
    assert(!emitted)
    emitted = true

    val l = new mutable.ArrayBuffer[AbstractInsnNode]()
    l ++= startup
    c.emit(l)

    val s = mutable.Set[AbstractInsnNode]()
    l.foreach { insn =>
      assert(!s.contains(insn))
      s += insn
    }

    l.foreach {
      case x: VarInsnNode =>
        MethodBuilder.popInsnRef(x) match {
          case Some(lr) =>
            lr.allocate(this, x)
          case None =>
            assert(x.`var` >= 0)
        }
      case x: IincInsnNode =>
        MethodBuilder.popInsnRef(x) match {
          case Some(lr) =>
            lr.allocate(this, x)
          case None =>
            assert(x.`var` >= 0)
        }
      case _ =>
    }

    mn.instructions.add(start)

    l.foreach(mn.instructions.add _)
    mn.instructions.add(new InsnNode(returnTypeInfo.returnOp))
    mn.instructions.add(end)
  }

  def invoke[T](args: Code[_]*): Code[T] =
    new Code[T] {
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        getArg[java.lang.Object](0).emit(il)
        args.foreach(_.emit(il))
        il += new MethodInsnNode(INVOKESPECIAL, fb.name, mname, descriptor, false)
      }
    }
}

trait DependentFunction[F >: Null <: AnyRef] extends FunctionBuilder[F] {
  var definedFields: ArrayBuilder[Growable[AbstractInsnNode] => Unit] = new ArrayBuilder(16)

  def addField[T : TypeInfo](value: Code[T]): ClassFieldRef[T] = {
    val cfr = newField[T]
    val add: (Growable[AbstractInsnNode]) => Unit = { (il: Growable[AbstractInsnNode]) =>
      il += new TypeInsnNode(CHECKCAST, name)
      value.emit(il)
      il += new FieldInsnNode(PUTFIELD, name, cfr.name, typeInfo[T].name)
    }
    definedFields += add
    cfr
  }

  def addField[T](value: Code[_], dummy: Boolean)(implicit ti: TypeInfo[T]): ClassFieldRef[T] = {
    val cfr = newField[T]
    val add: (Growable[AbstractInsnNode]) => Unit = { (il: Growable[AbstractInsnNode]) =>
      il += new TypeInsnNode(CHECKCAST, name)
      value.emit(il)
      il += new FieldInsnNode(PUTFIELD, name, cfr.name, typeInfo[T].name)
    }
    definedFields += add
    cfr
  }

  def newInstance()(implicit fct: ClassTag[F]): Code[F] = {
    val instance: Code[F] =
      new Code[F] {
        def emit(il: Growable[AbstractInsnNode]): Unit = {
          il += new TypeInsnNode(NEW, name)
          il += new InsnNode(DUP)
          il += new MethodInsnNode(INVOKESPECIAL, name, "<init>", "()V", false)
          il += new TypeInsnNode(CHECKCAST, classInfo[F].iname)
          definedFields.result().foreach { add =>
            il += new InsnNode(DUP)
            add(il)
          }
        }
      }
    instance
  }

  override def result(pw: Option[PrintWriter]): () => F =
    throw new UnsupportedOperationException("cannot call result() on a dependent function")

}

class DependentFunctionBuilder[F >: Null <: AnyRef : TypeInfo : ClassTag](
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated",
  initModule: ModuleBuilder = null
) extends FunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName, initModule = initModule) with DependentFunction[F]

class FunctionBuilder[F >: Null](
  val parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  val returnTypeInfo: MaybeGenericTypeInfo[_],
  val packageName: String = "is/hail/codegen/generated",
  namePrefix: String = null,
  initModule: ModuleBuilder = null)(implicit val interfaceTi: TypeInfo[F]) {
  import FunctionBuilder._

  val module: ModuleBuilder =
    if (initModule == null)
      new ModuleBuilder()
  else
      initModule

  val classBuilder: ClassBuilder[F] = new ClassBuilder[F](
    module,
    packageName + "/C" + Option(namePrefix).map(n => s"_${n}_").getOrElse("") + newUniqueID())

  val name: String = classBuilder.name

  private[this] val methodMemo: mutable.Map[Any, MethodBuilder] = mutable.HashMap.empty

  def getOrDefineMethod(suffix: String, key: Any, argsInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_])
    (f: MethodBuilder => Unit): MethodBuilder = {
    methodMemo.get(key) match {
      case Some(mb) => mb
      case None =>
        val mb = newMethod(suffix, argsInfo, returnInfo)
        f(mb)
        methodMemo(key) = mb
        mb
    }
  }

  private[this] lazy val _apply_method: MethodBuilder = {
    val m = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.base), returnTypeInfo.base)
    if (parameterTypeInfo.exists(_.isGeneric) || returnTypeInfo.isGeneric) {
      val generic = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.generic), returnTypeInfo.generic)
      classBuilder.addMethod(generic)
      generic.emit(
        new Code[Unit] {
          def emit(il: Growable[AbstractInsnNode]) {
            returnTypeInfo.castToGeneric(
              m.invoke(parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
                ti.castFromGeneric(generic.getArg(i + 1)(ti.generic))
              }: _*)).emit(il)
          }
        }
      )
    }
    m
  }

  def apply_method: MethodBuilder = _apply_method

  def newField[T: TypeInfo]: ClassFieldRef[T] = newField()

  def newField[T: TypeInfo](name: String = null): ClassFieldRef[T] =
    new ClassFieldRef[T](this, classBuilder.genField[T](name))

  def newLazyField[T: TypeInfo](setup: Code[T], name: String = null): LazyFieldRef[T] =
    new LazyFieldRef[T](this, name, setup)

  val lazyFieldMemo: mutable.Map[Any, LazyFieldRef[_]] = mutable.Map.empty

  def getOrDefineLazyField[T: TypeInfo](setup: Code[T], id: Any): LazyFieldRef[T] = {
    lazyFieldMemo.getOrElseUpdate(id, newLazyField[T](setup)).asInstanceOf[LazyFieldRef[T]]
  }

  def newLocal[T](implicit tti: TypeInfo[T]): LocalRef[T] = newLocal()

  def newLocal[T](name: String = null)(implicit tti: TypeInfo[T]): LocalRef[T] = apply_method.newLocal[T](name)

  def getArg[T](i: Int)(implicit tti: TypeInfo[T]): Settable[T] = apply_method.getArg[T](i)

  def emit(c: Code[_]) = apply_method.emit(c)

  def newMethod(suffix: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder = {
    val mb = new MethodBuilder(this, classBuilder.genName("m", suffix), argsInfo, returnInfo)
    classBuilder.addMethod(mb)
    mb
  }

  def newMethod(argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilder =
    newMethod("method", argsInfo, returnInfo)

  def newMethod[R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](), typeInfo[R])

  def newMethod[A: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D]), typeInfo[R])

  def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: MethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D], typeInfo[E]), typeInfo[R])

  def classAsBytes(print: Option[PrintWriter] = None): Array[Byte] = classBuilder.classAsBytes(print)

  classBuilder.addInterface(interfaceTi.iname)

  def result(print: Option[PrintWriter] = None): () => F = classBuilder.result(print)
}

class Function2Builder[A1 : TypeInfo, A2 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction2[A1, A2, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)
}

class Function3Builder[A1 : TypeInfo, A2 : TypeInfo, A3 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction3[A1, A2, A3, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)

  def arg3 = getArg[A3](3)
}

class Function4Builder[A1 : TypeInfo, A2 : TypeInfo, A3 : TypeInfo, A4 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction4[A1, A2, A3, A4, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3], GenericTypeInfo[A4]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)

  def arg3 = getArg[A3](3)

  def arg4 = getArg[A4](4)
}

class Function5Builder[A1 : TypeInfo, A2 : TypeInfo, A3 : TypeInfo, A4 : TypeInfo, A5 : TypeInfo, R : TypeInfo](name: String)
  extends FunctionBuilder[AsmFunction5[A1, A2, A3, A4, A5, R]](Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3], GenericTypeInfo[A4], GenericTypeInfo[A5]), GenericTypeInfo[R], namePrefix = name) {

  def arg1 = getArg[A1](1)

  def arg2 = getArg[A2](2)

  def arg3 = getArg[A3](3)

  def arg4 = getArg[A4](4)

  def arg5 = getArg[A5](5)
}
