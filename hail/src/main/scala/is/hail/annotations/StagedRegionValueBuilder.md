# StagedRegionValueBuilder
A [staged](CodeGeneration.md) implementation of [RegionValueBuilder](RegionValueBuilder.md).

## Use

[`new StagedRegionValueBuilder(fb: FunctionBuilder, rt: PType)`](https://github.com/akotlar/hail/blob/89f64981b8316187ae85c3b0ed8b745e6867ec96/hail/src/test/scala/is/hail/annotations/StagedRegionValueSuite.scala#L20)

FunctionBuilder instance can be created using either `val fb = FunctionBuilder.functionBuilder[Region, Int, Long]` or `val fb = EmitFunctionBuilder[Region, Long, Long]("foo")`
  * TODO: difference

After creating a FunctionBuilder instance, the next step is to `emit` some code that uses the StagedRegionValueBuilder instance.

```scala
val srvb = new StagedRegionValueBuilder(fb, rt)
fb.emit(
  Code(
    srvb.start(),
    srvb.addString(fb.getArg[String](2)),
    srvb.end()
  )
)
```

This "emit" function doesn't return anything. It takes a `Growable[AbstractInsNode]`, and performs some operation. Therea re many version of these operations, parameterized on the number of arguments, but all of them will call 

```scala
def emit(il: Growable[AbstractInsnNode]): Unit = {
  il += insn
}
```

This "+=" is a Scala method, not an operator. In Scala, this is allowed because a space is a valid alias for a `.` the typical method calling operator.

`+=` is a method on the [`Growable`](https://www.scala-lang.org/api/2.9.3/scala/collection/generic/Growable.html) trait, which extends `Clearable`. It adds an element to a Growable collection.
